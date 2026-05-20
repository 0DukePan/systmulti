"""
Stage 4 — Entropy Coding
Huffman coding (from scratch) + zlib on the full stream.
Pipeline:
  Encode: data → Huffman encode → zlib → .bin
  Decode: .bin → zlib → Huffman decode → data
"""

import zlib
import pickle
import struct
import os
import heapq
from collections import Counter


# ══════════════════════════════════════════════════════════════════
#  Huffman tree — implemented from scratch (academic requirement)
# ══════════════════════════════════════════════════════════════════

class _HNode:
    """A node in the Huffman binary tree."""
    __slots__ = ('freq', 'symbol', 'left', 'right')

    def __init__(self, freq, symbol=None, left=None, right=None):
        self.freq   = freq
        self.symbol = symbol
        self.left   = left
        self.right  = right

    # Comparison for the min-heap (by frequency)
    def __lt__(self, other):
        return self.freq < other.freq


def _build_huffman_tree(frequencies: dict) -> _HNode:
    """
    Build a Huffman tree from a symbol→frequency dict.
    Returns the root node.
    """
    if not frequencies:
        return None

    heap = [_HNode(freq, symbol=sym) for sym, freq in frequencies.items()]
    heapq.heapify(heap)

    if len(heap) == 1:
        # Edge case: only one symbol — wrap it so the tree has depth 1
        node = heapq.heappop(heap)
        root = _HNode(node.freq, left=node, right=_HNode(0))
        return root

    while len(heap) > 1:
        left  = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = _HNode(left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, merged)

    return heap[0]


def _build_codebook(node: _HNode, prefix: str = '',
                     codebook: dict = None) -> dict:
    """Traverse the tree and assign binary codes to each symbol."""
    if codebook is None:
        codebook = {}
    if node is None:
        return codebook
    if node.symbol is not None:   # leaf
        codebook[node.symbol] = prefix if prefix else '0'
    else:
        _build_codebook(node.left,  prefix + '0', codebook)
        _build_codebook(node.right, prefix + '1', codebook)
    return codebook


def _bits_to_bytes(bits: str) -> tuple[bytes, int]:
    """Pack a binary string into bytes, returns (bytes, padding_bits)."""
    padding = (8 - len(bits) % 8) % 8
    bits += '0' * padding
    byte_array = bytearray()
    for i in range(0, len(bits), 8):
        byte_array.append(int(bits[i:i+8], 2))
    return bytes(byte_array), padding


def _bytes_to_bits(data: bytes, padding: int) -> str:
    """Unpack bytes to binary string, removing padding."""
    bits = ''.join(f'{b:08b}' for b in data)
    if padding:
        bits = bits[:-padding]
    return bits


def huffman_encode(symbols: list) -> tuple[bytes, dict, int]:
    """
    Huffman-encode a list of symbols (any hashable type).

    Returns
    -------
    (encoded_bytes, codebook, padding)
    """
    if not symbols:
        return b'', {}, 0

    freq = Counter(symbols)
    root = _build_huffman_tree(freq)
    codebook = _build_codebook(root)

    bits = ''.join(codebook[s] for s in symbols)
    encoded, padding = _bits_to_bytes(bits)
    return encoded, codebook, padding


def huffman_decode(encoded: bytes, codebook: dict,
                   padding: int, n_symbols: int = None) -> list:
    """
    Huffman-decode bytes back to the original symbol list.

    Parameters
    ----------
    encoded   : bytes produced by huffman_encode
    codebook  : symbol → bit-string dict
    padding   : number of padding bits at the end
    n_symbols : expected number of decoded symbols (for early stop)
    """
    if not codebook:
        return []

    # Build reverse code → symbol map
    rev = {v: k for k, v in codebook.items()}

    bits = _bytes_to_bits(encoded, padding)
    symbols = []
    buf = ''
    for bit in bits:
        buf += bit
        if buf in rev:
            symbols.append(rev[buf])
            buf = ''
            if n_symbols is not None and len(symbols) >= n_symbols:
                break
    return symbols


# ══════════════════════════════════════════════════════════════════
#  Binary file format
# ══════════════════════════════════════════════════════════════════
#
#  Header (20 bytes):
#    4B  magic  "M4SP"
#    4B  width  (uint32 big-endian)
#    4B  height (uint32 big-endian)
#    4B  n_frames (uint32)
#    2B  gop    (uint16)
#    2B  qf     (uint16)
#
#  Body: zlib(pickle(frame_data)) — Huffman is applied per-frame
#        internally inside each frame's encoded struct.

MAGIC = b'M4SP'


def compress_stream(frame_data: list, out_path: str,
                    width: int, height: int,
                    gop: int, qf: int) -> int:
    """
    Serialize, Huffman-compress, then zlib-compress the frame list.

    Each frame's RLE data and motion vectors are Huffman-coded
    before being bundled and passed through zlib for a second
    lossless pass.
    """
    processed = _huffman_compress_frames(frame_data)

    raw = pickle.dumps(processed)
    compressed = zlib.compress(raw, level=9)

    header = MAGIC + struct.pack('>IIIHH', width, height,
                                  len(frame_data), gop, qf)
    with open(out_path, 'wb') as f:
        f.write(header)
        f.write(compressed)

    return os.path.getsize(out_path)


def decompress_stream(bin_path: str) -> tuple[list, dict]:
    """
    Read a .bin file, zlib-decompress, then Huffman-decode each frame.
    """
    with open(bin_path, 'rb') as f:
        header_raw = f.read(20)
        compressed = f.read()

    if header_raw[:4] != MAGIC:
        raise ValueError(f"Bad magic: {header_raw[:4]}. Not an M4SP file.")

    width, height, n_frames, gop, qf = struct.unpack('>IIIHH', header_raw[4:])

    raw = zlib.decompress(compressed)
    processed = pickle.loads(raw)
    frame_data = _huffman_decompress_frames(processed)

    return frame_data, {
        'width': width, 'height': height,
        'n_frames': n_frames, 'gop': gop, 'qf': qf,
    }


# ══════════════════════════════════════════════════════════════════
#  Per-frame Huffman helpers
# ══════════════════════════════════════════════════════════════════

def _huffman_compress_frames(frames: list) -> list:
    """Apply Huffman coding to RLE data and motion vectors per frame."""
    out = []
    for fd in frames:
        ftype = fd['type']
        entry = {'type': ftype}

        # ── Y channel ───────────────────────────────────────────
        entry['Y']  = _huff_encode_channel(fd['Y'],  ftype)
        # ── Chroma (always Intra-coded) ──────────────────────────
        entry['Cb'] = _huff_encode_channel(fd['Cb'], 'I')
        entry['Cr'] = _huff_encode_channel(fd['Cr'], 'I')
        out.append(entry)
    return out


def _huffman_decompress_frames(frames: list) -> list:
    out = []
    for fd in frames:
        ftype = fd['type']
        entry = {'type': ftype}
        entry['Y']  = _huff_decode_channel(fd['Y'],  ftype)
        entry['Cb'] = _huff_decode_channel(fd['Cb'], 'I')
        entry['Cr'] = _huff_decode_channel(fd['Cr'], 'I')
        out.append(entry)
    return out


def _huff_encode_channel(ch_data: dict, ftype: str) -> dict:
    """Huffman-encode the RLE list of a single channel."""
    result = dict(ch_data)   # shallow copy (preserves shape, qf, chroma)

    if ftype == 'I':
        # Flatten all (run, value) pairs from the RLE blocks
        rle_list = ch_data['rle']
        flat = []
        for block_rle in rle_list:
            for pair in block_rle:
                flat.append(pair)

        enc, cb, pad = huffman_encode(flat)
        result['rle_huff']  = enc
        result['rle_cb']    = cb
        result['rle_pad']   = pad
        result['rle_nblk']  = len(rle_list)
        result['rle_npair'] = len(flat)
        del result['rle']

    else:
        # P-frame: Huffman-encode motion vectors (already differential)
        # and the residual RLE
        mv_list = ch_data['mv']
        mv_flat = [v for pair in mv_list for v in pair]
        enc_mv, cb_mv, pad_mv = huffman_encode(mv_flat)
        result['mv_huff']  = enc_mv
        result['mv_cb']    = cb_mv
        result['mv_pad']   = pad_mv
        result['mv_n']     = len(mv_list)
        del result['mv']

        # Encode residual RLE
        res_enc = _huff_encode_channel(ch_data['residual_encoded'], 'I')
        result['residual_encoded'] = res_enc

    return result


def _huff_decode_channel(ch_data: dict, ftype: str) -> dict:
    """Huffman-decode a single channel back to original structure."""
    result = dict(ch_data)

    if ftype == 'I':
        flat = huffman_decode(ch_data['rle_huff'], ch_data['rle_cb'],
                              ch_data['rle_pad'],  ch_data['rle_npair'])
        # Split flat pairs list back into per-block lists
        blocks = []
        idx = 0
        while idx < len(flat):
            block = []
            while idx < len(flat):
                pair = flat[idx]
                block.append(pair)
                idx += 1
                if pair == (0, 0):   # EOB marker
                    break
            blocks.append(block)

        result['rle'] = blocks
        for k in ('rle_huff','rle_cb','rle_pad','rle_nblk','rle_npair'):
            result.pop(k, None)

    else:
        mv_flat = huffman_decode(ch_data['mv_huff'], ch_data['mv_cb'],
                                 ch_data['mv_pad'],  ch_data['mv_n'] * 2)
        result['mv'] = [(mv_flat[i], mv_flat[i+1])
                        for i in range(0, len(mv_flat), 2)]

        for k in ('mv_huff','mv_cb','mv_pad','mv_n'):
            result.pop(k, None)

        result['residual_encoded'] = _huff_decode_channel(
            ch_data['residual_encoded'], 'I')

    return result
