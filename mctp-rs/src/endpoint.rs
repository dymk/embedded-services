//! `MctpEndpoint<M>` — sync, packet-oriented owning wrapper over
//! [`MctpPacketContext`] + an internal [`MctpSequenceNumber`].
//!
//! `send` drives serialization through a per-packet `FnMut(&[u8])` closure
//! whose `&[u8]` borrow does not escape the call. `recv_packet` returns an
//! owned [`RecvMessage`] (no `&self` lifetime escape) — closing the v1.5
//! "borrowed-from-self" lifetime trap (PITFALLS CP-5) at the API surface.
//!
//! See `33-CONTEXT.md § "API style decision (chosen by planner)"` in the
//! planning repo for the full rationale on the sync, packet-oriented shape
//! (versus byte-stream / async alternatives).

use crate::{
    EndpointId, MctpMessageTag, MctpPacketContext, MctpPacketError, MctpReplyContext, MctpSequenceNumber,
    error::MctpPacketResult, medium::MctpMedium, message_type::MctpMessageTrait,
};

/// Maximum decoded body bytes carried by a single [`RecvMessage`].
///
/// Matches the `BUF_SIZE = 256` used by `uart-service` (the only consumer in
/// the v1.8 path). A free module-scope const is used (not an associated
/// const through `MctpEndpoint::<'_, M>`) so callers don't need
/// `feature(generic_const_exprs)`.
pub const MAX_MESSAGE_BODY: usize = 256;

/// Owning MCTP endpoint: composes [`MctpPacketContext`] with an internal
/// [`MctpSequenceNumber`] counter and a fixed source [`EndpointId`].
///
/// Pure additive: composes over existing public API; does not extend any
/// existing trait.
pub struct MctpEndpoint<'buf, M: MctpMedium> {
    ctx: MctpPacketContext<'buf, M>,
    seq: MctpSequenceNumber,
    source_endpoint_id: EndpointId,
}

/// Owned message returned from [`MctpEndpoint::recv_packet`] when a full
/// MCTP message has been assembled.
///
/// `body` is owned (no borrow of the endpoint's assembly buffer escapes);
/// `reply_context` reuses the existing [`MctpReplyContext`] shape (no
/// parallel struct duplication).
pub struct RecvMessage<M: MctpMedium> {
    pub reply_context: MctpReplyContext<M>,
    /// Owned decoded body bytes (post message-type byte).
    pub body: heapless::Vec<u8, MAX_MESSAGE_BODY>,
    /// Message-type byte (e.g., `0x7E` for VendorDefinedPci).
    pub message_type: u8,
    /// Optional message-integrity-check byte (present iff IC bit set).
    pub message_integrity_check: Option<u8>,
}

impl<'buf, M: MctpMedium> MctpEndpoint<'buf, M> {
    /// Construct an endpoint over `medium`, using `assembly_buf` as the
    /// internal packet-assembly buffer. The sequence-number counter starts
    /// at 0; `source_endpoint_id` is used for outgoing messages.
    pub fn new(medium: M, assembly_buf: &'buf mut [u8], source_endpoint_id: EndpointId) -> Self {
        Self {
            ctx: MctpPacketContext::new(medium, assembly_buf),
            seq: MctpSequenceNumber::new(0),
            source_endpoint_id,
        }
    }

    /// Send `payload` to `dst`, invoking `write_packet` once per emitted
    /// wire-ready packet.
    ///
    /// The `&[u8]` slice handed to `write_packet` borrows the endpoint's
    /// assembly buffer; the borrow is bounded by the closure call and
    /// does NOT escape `send` (closes v1.5 PITFALLS CP-5).
    ///
    /// `Err(())` from `write_packet` aborts serialization with
    /// [`MctpPacketError::WriteAborted`]; callers that need to surface
    /// their own I/O error should capture it via `&mut Option<E>` closed
    /// over by the closure.
    pub fn send<P>(
        &'buf mut self,
        dst: EndpointId,
        message_tag: MctpMessageTag,
        medium_reply_ctx: M::ReplyContext,
        payload: (P::Header, P),
        mut write_packet: impl FnMut(&[u8]) -> Result<(), ()>,
    ) -> MctpPacketResult<(), M>
    where
        P: MctpMessageTrait<'buf>,
    {
        let reply_context = MctpReplyContext {
            destination_endpoint_id: dst,
            source_endpoint_id: self.source_endpoint_id,
            packet_sequence_number: self.seq,
            message_tag,
            medium_context: medium_reply_ctx,
        };

        let mut state = self.ctx.serialize_packet(reply_context, payload)?;
        while let Some(packet_result) = state.next() {
            let packet = packet_result?;
            write_packet(packet).map_err(|_| MctpPacketError::WriteAborted)?;
        }

        // Sequence number advances once per outgoing message (the
        // per-packet `seq.inc()` driven inside SerializePacketState only
        // mutates a copy; advance the durable counter here).
        self.seq.inc();
        Ok(())
    }

    /// Feed one already-framed packet into the assembler. Returns
    /// `Ok(Some(_))` when a complete message has been assembled (and the
    /// returned [`RecvMessage`] carries OWNED bytes — no `&self` borrow
    /// escapes), `Ok(None)` for partial/intermediate packets.
    ///
    /// The packet-framing contract matches
    /// [`MctpPacketContext::deserialize_packet`] verbatim — caller is
    /// responsible for delivering one already-framed wire packet per call.
    pub fn recv_packet(&mut self, packet: &[u8]) -> MctpPacketResult<Option<RecvMessage<M>>, M> {
        let Some(message) = self.ctx.deserialize_packet(packet)? else {
            return Ok(None);
        };
        // Copy borrowed body bytes into an owned heapless Vec before the
        // borrow on &self.ctx expires.
        let body_slice = message.message_buffer.rest;
        let body = heapless::Vec::from_slice(body_slice)
            .map_err(|_| MctpPacketError::HeaderParseError("recv_packet: body exceeds MAX_MESSAGE_BODY"))?;
        Ok(Some(RecvMessage {
            reply_context: message.reply_context,
            body,
            message_type: message.message_buffer.message_type,
            message_integrity_check: message.message_integrity_check,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        VendorDefinedPci, VendorDefinedPciHeader,
        buffer_encoding::{EncodingDecoder, EncodingEncoder, PassthroughEncoding},
        medium::MctpMediumFrame,
    };

    /// Minimal `no_std`-compatible mock medium: zero header/trailer,
    /// PassthroughEncoding (no byte-stuffing). `Vec<Vec<u8>>` channel is
    /// supplied by the test, not the medium itself, so the medium stays
    /// trivially `Copy + Clone`.
    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    struct MockMedium {
        mtu: usize,
    }

    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    struct MockMediumFrame {
        packet_size: usize,
    }

    impl MctpMedium for MockMedium {
        type Frame = MockMediumFrame;
        type Error = &'static str;
        type ReplyContext = ();
        type Encoding = PassthroughEncoding;

        fn max_message_body_size(&self) -> usize {
            self.mtu
        }

        fn deserialize<'buf>(
            &self,
            packet: &'buf [u8],
        ) -> MctpPacketResult<(Self::Frame, EncodingDecoder<'buf, Self::Encoding>), Self> {
            Ok((
                MockMediumFrame {
                    packet_size: packet.len(),
                },
                EncodingDecoder::new(packet),
            ))
        }

        fn serialize<'buf, F>(
            &self,
            _: Self::ReplyContext,
            buffer: &'buf mut [u8],
            message_writer: F,
        ) -> MctpPacketResult<&'buf [u8], Self>
        where
            F: for<'a> FnOnce(&mut EncodingEncoder<'a, Self::Encoding>) -> MctpPacketResult<(), Self>,
        {
            let max = self.mtu.min(buffer.len());
            let n = {
                let mut encoder = EncodingEncoder::<Self::Encoding>::new(&mut buffer[..max]);
                message_writer(&mut encoder)?;
                encoder.wire_position()
            };
            Ok(&buffer[..n])
        }
    }

    impl MctpMediumFrame<MockMedium> for MockMediumFrame {
        fn packet_size(&self) -> usize {
            self.packet_size
        }
        fn reply_context(&self) {}
    }

    /// Round-trip a 32-byte payload through `send -> MockMedium ->
    /// recv_packet`, asserting the recovered body bytes equal the
    /// original. Exercises the owned-bytes invariant (the recovered
    /// `RecvMessage::body` outlives both endpoints' assembly buffers).
    #[test]
    fn round_trip_single_packet() {
        extern crate std;
        use std::vec::Vec as StdVec;

        let payload: [u8; 32] = [0xAA; 32];

        // Send side: collect emitted packets into an owned Vec<Vec<u8>>.
        let mut send_buf = [0u8; 1024];
        let mut tx = MctpEndpoint::new(MockMedium { mtu: 256 }, &mut send_buf, EndpointId::Id(0x10));
        let mut packets: StdVec<StdVec<u8>> = StdVec::new();
        tx.send(
            EndpointId::Id(0x20),
            MctpMessageTag::try_from(3).unwrap(),
            (),
            (VendorDefinedPciHeader(0x1234), VendorDefinedPci(&payload)),
            |pkt: &[u8]| {
                packets.push(pkt.to_vec());
                Ok(())
            },
        )
        .expect("send ok");
        assert!(!packets.is_empty(), "send emitted at least one packet");

        // Recv side: feed packets back through a fresh endpoint.
        let mut recv_buf = [0u8; 1024];
        let mut rx = MctpEndpoint::new(MockMedium { mtu: 256 }, &mut recv_buf, EndpointId::Id(0x20));
        let mut last: Option<RecvMessage<MockMedium>> = None;
        for pkt in &packets {
            let r = rx.recv_packet(pkt).expect("recv_packet ok");
            if r.is_some() {
                last = r;
            }
        }
        let msg = last.expect("recv assembled a complete message");
        assert_eq!(msg.message_type, VendorDefinedPci::MESSAGE_TYPE);
        // body = vendor-defined-pci header (2 bytes BE: 0x12, 0x34) ++ payload
        assert_eq!(&msg.body[..2], &[0x12, 0x34]);
        assert_eq!(&msg.body[2..], &payload[..]);
    }
}
