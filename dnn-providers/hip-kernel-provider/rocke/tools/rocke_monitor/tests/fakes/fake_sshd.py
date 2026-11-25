#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""An SSH server for the tests, on Python's standard library alone: a stand-in host that OpenSSH's own ssh,
ssh-keyscan and rsync reach (the image test's web-host stage, tests/container/stage_web_host.py; its own tests,
tests/test_fake_sshd.py). It speaks SSH 2 with one algorithm of each kind, among those OpenSSH's client offers: the key
exchange curve25519-sha256, the host key ssh-ed25519, OpenSSH's cipher chacha20-poly1305 (which carries its own MAC)
and no compression; one key exchange per connection. It lets one login in, by one public key (ssh-ed25519), and
runs each command it is asked to run with `sh -c`, as the login's shell would: in the login's home, its standard
input, output and errors over the session, then its exit status. Nothing else: no shell, terminal, forwarding or
subsystem, each refused, and one session per connection.

    python3 fake_sshd.py --host-key FILE --authorized-key FILE --user NAME --home DIR [--address 0.0.0.0]
        [--port 22] [--path PATH] [--log FILE] [--ready FILE]

--host-key: the host's own key, an OpenSSH private key with no passphrase (`ssh-keygen -t ed25519 -N ""`).
--authorized-key: the login's public key, a line of authorized_keys (ssh-keygen's .pub). Each is read again at each
connection, so a test may change either while the server runs. --user: the login's name, the only one let in.
--home: the login's home, where each command runs, with HOME, USER and LOGNAME set, on PATH (--path; the system's
folders when none is given). --log: what each connection did, a JSON line each, with the connection's number and the
client's address: connect, kex (the algorithms agreed and the host key's SHA256 fingerprint), auth (the login, whether
the request was signed, whether it was let in), exec (the command), exit (its status) and closed (why); never a key.
--ready: written with the port once the server listens (--port 0: a free one).

It is a test's stand-in, never a server to use: it checks what OpenSSH's client sends no further than the tests need.
"""

import argparse
import base64
import hashlib
import hmac
import itertools
import json
import os
import pathlib
import queue
import signal
import socket
import socketserver
import struct
import subprocess
import threading
import time

VERSION = b"SSH-2.0-rkmonTestSSHD_1.0"
KEX = ("curve25519-sha256",)
KEY_TYPE = "ssh-ed25519"  # the host key's, and the login's
# OpenSSH's name for its chacha20-poly1305, in two parts: the privacy check reads it as an email address
CIPHER = "chacha20-poly1305@" "openssh.com"
MAC = "hmac-sha2-256"  # listed only: the cipher carries its own MAC, and OpenSSH negotiates none beside it
# what the client may send on the session before it waits for this side to read it
WINDOW = 1 << 18
MAX_PACKET = 1 << 15  # the most data in one message, each way
MAX_LENGTH = 1 << 18  # the longest packet taken
IDLE_S = 600  # a connection quiet this long is closed
BASE_POINT = (9).to_bytes(32, "little")
PRIVATE_KEY_MAGIC = b"openssh-key-v1\0"

# Message numbers (RFC 4250)
DISCONNECT, IGNORE, UNIMPLEMENTED, DEBUG = 1, 2, 3, 4
SERVICE_REQUEST, SERVICE_ACCEPT = 5, 6
KEXINIT, NEWKEYS = 20, 21
KEX_ECDH_INIT, KEX_ECDH_REPLY = 30, 31
USERAUTH_REQUEST, USERAUTH_FAILURE, USERAUTH_SUCCESS, USERAUTH_PK_OK = 50, 51, 52, 60
GLOBAL_REQUEST, REQUEST_FAILURE = 80, 82
CHANNEL_OPEN, CHANNEL_OPEN_CONFIRMATION, CHANNEL_OPEN_FAILURE = 90, 91, 92
CHANNEL_WINDOW_ADJUST, CHANNEL_DATA, CHANNEL_EXTENDED_DATA = 93, 94, 95
CHANNEL_EOF, CHANNEL_CLOSE, CHANNEL_REQUEST = 96, 97, 98
CHANNEL_SUCCESS, CHANNEL_FAILURE = 99, 100
CHANNEL_MESSAGES = (
    CHANNEL_WINDOW_ADJUST,
    CHANNEL_DATA,
    CHANNEL_EXTENDED_DATA,
    CHANNEL_EOF,
    CHANNEL_CLOSE,
    CHANNEL_REQUEST,
)
# DISCONNECT's reasons (RFC 4250)
PROTOCOL_ERROR, KEY_EXCHANGE_FAILED, MAC_ERROR = 2, 3, 5
SERVICE_NOT_AVAILABLE, NO_MORE_AUTH_METHODS = 7, 14

# --- the arithmetic: X25519 (RFC 7748), Ed25519 (RFC 8032), ChaCha20 and Poly1305 (RFC 8439) --------------------------

P = 2**255 - 19
L = 2**252 + 27742317777372353535851937790883648493  # the order of Ed25519's base point
_D = -121665 * pow(121666, P - 2, P) % P
_SQRT_M1 = pow(2, (P - 1) // 4, P)
_M32 = 0xFFFFFFFF
# ChaCha20's quarter rounds of one double round: the columns, then the diagonals
_ROUNDS = (
    (0, 4, 8, 12),
    (1, 5, 9, 13),
    (2, 6, 10, 14),
    (3, 7, 11, 15),
    (0, 5, 10, 15),
    (1, 6, 11, 12),
    (2, 7, 8, 13),
    (3, 4, 9, 14),
)
_SIGMA = struct.unpack("<4I", b"expand 32-byte k")


def x25519(scalar: bytes, u: bytes) -> bytes:
    """X25519 of the 32-byte SCALAR and the u-coordinate U, 32 bytes each way (RFC 7748, 5)."""
    k = bytearray(scalar)
    k[0] &= 248
    k[31] &= 127
    k[31] |= 64
    n = int.from_bytes(bytes(k), "little")
    x1 = int.from_bytes(u, "little") & ((1 << 255) - 1)
    x2, z2, x3, z3, swap = 1, 0, x1, 1, 0
    for t in range(254, -1, -1):
        bit = (n >> t) & 1
        if swap ^ bit:
            x2, x3, z2, z3 = x3, x2, z3, z2
        swap = bit
        a, b = x2 + z2, x2 - z2
        aa, bb = a * a % P, b * b % P
        e = aa - bb
        c, d = x3 + z3, x3 - z3
        da, cb = d * a % P, c * b % P
        x3 = (da + cb) ** 2 % P
        z3 = x1 * (da - cb) ** 2 % P
        x2 = aa * bb % P
        z2 = e * (aa + 121665 * e) % P
    if swap:
        x2, z2 = x3, z3
    return (x2 * pow(z2, P - 2, P) % P).to_bytes(32, "little")


def _add(p1: tuple, p2: tuple) -> tuple:
    """The sum of two points of Ed25519's curve, in extended coordinates (RFC 8032, 5.1.4)."""
    x1, y1, z1, t1 = p1
    x2, y2, z2, t2 = p2
    a = (y1 - x1) * (y2 - x2) % P
    b = (y1 + x1) * (y2 + x2) % P
    c = 2 * t1 * t2 * _D % P
    d = 2 * z1 * z2 % P
    e, f, g, h = b - a, d - c, d + c, b + a
    return (e * f % P, g * h % P, f * g % P, e * h % P)


def _times(n: int, point: tuple) -> tuple:
    out = (0, 1, 1, 0)
    while n:
        if n & 1:
            out = _add(out, point)
        point = _add(point, point)
        n >>= 1
    return out


def _x_of(y: int, sign: int):
    """The x-coordinate of the point with Y and the sign SIGN (RFC 8032, 5.1.3); None for no point."""
    if y >= P:
        return None
    xx = (y * y - 1) * pow(_D * y * y + 1, P - 2, P) % P
    if xx == 0:
        return None if sign else 0
    x = pow(xx, (P + 3) // 8, P)
    if (x * x - xx) % P:
        x = x * _SQRT_M1 % P
    if (x * x - xx) % P:
        return None
    return P - x if (x & 1) != sign else x


_GY = 4 * pow(5, P - 2, P) % P
_GX = _x_of(_GY, 0)
_G = (_GX, _GY, 1, _GX * _GY % P)


def _encode(point: tuple) -> bytes:
    x, y, z, _ = point
    zi = pow(z, P - 2, P)
    x, y = x * zi % P, y * zi % P
    return (y | ((x & 1) << 255)).to_bytes(32, "little")


def _decode(data: bytes):
    if len(data) != 32:
        return None
    y = int.from_bytes(data, "little")
    sign, y = y >> 255, y & ((1 << 255) - 1)
    x = _x_of(y, sign)
    return None if x is None else (x, y, 1, x * y % P)


def _expand(seed: bytes) -> tuple:
    h = hashlib.sha512(seed).digest()
    a = int.from_bytes(h[:32], "little")
    return (a & ((1 << 254) - 8)) | (1 << 254), h[32:]


def _scalar(data: bytes) -> int:
    return int.from_bytes(hashlib.sha512(data).digest(), "little") % L


def ed25519_public(seed: bytes) -> bytes:
    """The public key of the 32-byte private key SEED (RFC 8032, 5.1.5)."""
    return _encode(_times(_expand(seed)[0], _G))


def ed25519_sign(seed: bytes, message: bytes) -> bytes:
    """MESSAGE's signature by the private key SEED, 64 bytes (RFC 8032, 5.1.6)."""
    a, prefix = _expand(seed)
    public = _encode(_times(a, _G))
    r = _scalar(prefix + message)
    big_r = _encode(_times(r, _G))
    s = (r + _scalar(big_r + public + message) * a) % L
    return big_r + s.to_bytes(32, "little")


def ed25519_verify(public: bytes, message: bytes, signature: bytes) -> bool:
    """Whether SIGNATURE is MESSAGE's by the public key PUBLIC (RFC 8032, 5.1.7)."""
    if len(signature) != 64:
        return False
    a, r = _decode(public), _decode(signature[:32])
    s = int.from_bytes(signature[32:], "little")
    if a is None or r is None or s >= L:
        return False
    left = _times(s, _G)
    right = _add(r, _times(_scalar(signature[:32] + public + message), a))
    return (left[0] * right[2] - right[0] * left[2]) % P == 0 and (
        left[1] * right[2] - right[1] * left[2]
    ) % P == 0


def chacha20_block(key: bytes, words: bytes) -> bytes:
    """ChaCha20's block of the 32-byte KEY and WORDS, the 16 bytes of its last four words (the block counter and the
    nonce, as the variant has them): 64 bytes of key stream (RFC 8439, 2.3)."""
    s = (*_SIGMA, *struct.unpack("<8I", key), *struct.unpack("<4I", words))
    x = list(s)
    for _ in range(10):
        for a, b, c, d in _ROUNDS:
            x[a] = (x[a] + x[b]) & _M32
            v = x[d] ^ x[a]
            x[d] = ((v << 16) & _M32) | (v >> 16)
            x[c] = (x[c] + x[d]) & _M32
            v = x[b] ^ x[c]
            x[b] = ((v << 12) & _M32) | (v >> 20)
            x[a] = (x[a] + x[b]) & _M32
            v = x[d] ^ x[a]
            x[d] = ((v << 8) & _M32) | (v >> 24)
            x[c] = (x[c] + x[d]) & _M32
            v = x[b] ^ x[c]
            x[b] = ((v << 7) & _M32) | (v >> 25)
    return struct.pack("<16I", *[(x[i] + s[i]) & _M32 for i in range(16)])


def poly1305(key: bytes, message: bytes) -> bytes:
    """MESSAGE's Poly1305 tag by the 32-byte one-time KEY, 16 bytes (RFC 8439, 2.5)."""
    r = int.from_bytes(key[:16], "little") & 0x0FFFFFFC0FFFFFFC0FFFFFFC0FFFFFFF
    s = int.from_bytes(key[16:32], "little")
    p = (1 << 130) - 5
    acc = 0
    for i in range(0, len(message), 16):
        acc = (acc + int.from_bytes(message[i : i + 16] + b"\x01", "little")) * r % p
    return ((acc + s) & ((1 << 128) - 1)).to_bytes(16, "little")


def _stream(key: bytes, seq: int, counter: int, size: int) -> bytes:
    """SIZE bytes of the key stream the cipher (CIPHER) takes for the packet SEQ (its nonce, as a big-endian 64-bit
    number), from the block COUNTER on."""
    nonce = struct.pack(">Q", seq)
    blocks = (size + 63) // 64
    out = b"".join(
        chacha20_block(key, struct.pack("<Q", counter + i) + nonce)
        for i in range(blocks)
    )
    return out[:size]


def _xor(data: bytes, stream: bytes) -> bytes:
    n = len(data)
    return (
        int.from_bytes(data, "little") ^ int.from_bytes(stream[:n], "little")
    ).to_bytes(n, "little")


# --- the wire ---------------------------------------------------------------------------------------------------------


class SSHError(Exception):
    """Why a connection ends, from this side."""


def _u32(n: int) -> bytes:
    return struct.pack(">I", n)


def _string(data) -> bytes:
    if isinstance(data, str):
        data = data.encode()
    return _u32(len(data)) + data


def _mpint(n: int) -> bytes:
    """A positive N as SSH's mpint (RFC 4251, 5): big-endian, a leading 0 byte where the top bit is set."""
    return _string(n.to_bytes((n.bit_length() + 8) // 8, "big") if n else b"")


def _message(kind: int, *parts: bytes) -> bytes:
    return bytes([kind]) + b"".join(parts)


class Reader:
    """The fields of one message, in order."""

    def __init__(self, data: bytes):
        self.data, self.at = data, 0

    def take(self, n: int) -> bytes:
        if self.at + n > len(self.data):
            raise SSHError("a message cut short")
        out = self.data[self.at : self.at + n]
        self.at += n
        return out

    def byte(self) -> int:
        return self.take(1)[0]

    def boolean(self) -> bool:
        return self.byte() != 0

    def u32(self) -> int:
        return struct.unpack(">I", self.take(4))[0]

    def string(self) -> bytes:
        return self.take(self.u32())

    def text(self) -> str:
        return self.string().decode("utf-8", "replace")

    def names(self) -> list:
        text = self.text()
        return text.split(",") if text else []


def fingerprint(blob: bytes) -> str:
    """A public key's SHA256 fingerprint, as ssh-keygen -l shows it."""
    return "SHA256:" + base64.b64encode(hashlib.sha256(blob).digest()).decode().rstrip(
        "="
    )


def key_blob(public: bytes) -> bytes:
    """An ssh-ed25519 public key as SSH sends it."""
    return _string(KEY_TYPE) + _string(public)


def read_private_key(path) -> tuple:
    """(seed, public key) of the OpenSSH private key at PATH, an ssh-ed25519 key with no passphrase."""
    lines = pathlib.Path(path).read_text(encoding="ascii").strip().splitlines()
    data = base64.b64decode("".join(x.strip() for x in lines[1:-1]))
    if not data.startswith(PRIVATE_KEY_MAGIC):
        raise SSHError(f"{path} holds no OpenSSH private key")
    r = Reader(data[len(PRIVATE_KEY_MAGIC) :])
    cipher, _kdf, _options, count = r.string(), r.string(), r.string(), r.u32()
    if cipher != b"none" or count != 1:
        raise SSHError(f"{path} holds a key with a passphrase, or more than one key")
    r.string()  # the public key, again below
    k = Reader(r.string())
    if k.u32() != k.u32():
        raise SSHError(f"{path}: the key's check numbers differ")
    if k.text() != KEY_TYPE:
        raise SSHError(f"{path} holds no {KEY_TYPE} key")
    public, secret = k.string(), k.string()
    return secret[:32], public


def read_public_key(path) -> bytes:
    """The key blob of the first ssh-ed25519 line of the authorized_keys file PATH."""
    for line in pathlib.Path(path).read_text(encoding="ascii").splitlines():
        f = line.split()
        if len(f) >= 2 and f[0] == KEY_TYPE:
            return base64.b64decode(f[1])
    raise SSHError(f"{path} holds no {KEY_TYPE} key")


class Connection:
    """One client's connection: the version lines, the key exchange, the login, then one session."""

    def __init__(self, sock, options, event):
        self.sock, self.o, self.event = (
            sock,
            options,
            event,
        )  # EVENT(what, **fields): a line of the log
        self.rfile = sock.makefile("rb")
        self.lock = (
            threading.Lock()
        )  # one packet out at a time, its sequence number with it
        self.seq_in = self.seq_out = 0
        self.keys_in = self.keys_out = (
            None  # (main key, length key) once NEWKEYS is past
        )
        self.session_id = b""
        self.session = None

    # --- packets ------------------------------------------------------------------------------------------------------
    def read(self, n: int) -> bytes:
        data = self.rfile.read(n)
        if len(data) < n:
            raise EOFError
        return data

    def receive(self) -> bytes:
        """The next packet's payload."""
        if self.keys_in is None:
            head = self.read(4)
            (length,) = struct.unpack(">I", head)
            if not 5 <= length <= MAX_LENGTH or (length + 4) % 8:
                raise SSHError(f"a packet of length {length}")
            body = self.read(length)
        else:
            main, size = self.keys_in
            sealed = self.read(4)
            (length,) = struct.unpack(
                ">I", _xor(sealed, _stream(size, self.seq_in, 0, 4))
            )
            if not 5 <= length <= MAX_LENGTH or length % 8:
                raise SSHError(f"a packet of length {length}")
            ciphered, tag = self.read(length), self.read(16)
            mac_key = _stream(main, self.seq_in, 0, 32)
            if not hmac.compare_digest(poly1305(mac_key, sealed + ciphered), tag):
                raise SSHError("a packet whose MAC is wrong")
            body = _xor(ciphered, _stream(main, self.seq_in, 1, length))
        self.seq_in = (self.seq_in + 1) & _M32
        padding = body[0]
        if padding < 4 or length - padding - 1 < 1:  # a message is one byte at least
            raise SSHError(f"a packet with {padding} bytes of padding")
        return body[1 : length - padding]

    def send(self, payload: bytes) -> None:
        with self.lock:
            # Padding to a whole number of 8-byte blocks: the length field with the rest before the keys, without it
            # after, as the cipher seals the length on its own.
            whole = (1 if self.keys_out else 5) + len(payload)
            padding = 8 - whole % 8
            padding += 8 if padding < 4 else 0
            body = bytes([padding]) + payload + os.urandom(padding)
            head = _u32(len(body))
            if self.keys_out is None:
                packet = head + body
            else:
                main, size = self.keys_out
                sealed = _xor(head, _stream(size, self.seq_out, 0, 4))
                ciphered = _xor(body, _stream(main, self.seq_out, 1, len(body)))
                mac_key = _stream(main, self.seq_out, 0, 32)
                packet = sealed + ciphered + poly1305(mac_key, sealed + ciphered)
            self.sock.sendall(packet)
            self.seq_out = (self.seq_out + 1) & _M32

    def disconnect(self, reason: int, text: str) -> None:
        try:
            self.send(_message(DISCONNECT, _u32(reason), _string(text), _string("")))
        except OSError:
            pass
        raise SSHError(text)

    def expect(self, kind: int) -> bytes:
        """The next message of the kind KIND, past those that say nothing; a DISCONNECT ends the connection."""
        while True:
            payload = self.receive()
            if payload[0] in (IGNORE, DEBUG, UNIMPLEMENTED):
                continue
            if payload[0] == DISCONNECT:
                raise EOFError
            if payload[0] != kind:
                self.disconnect(
                    PROTOCOL_ERROR, f"message {payload[0]} where {kind} was due"
                )
            return payload

    # --- the connection, start to end ---------------------------------------------------------------------------------
    def run(self) -> None:
        why = "the client closed it"
        try:
            self.serve()
        except EOFError:
            pass
        except (SSHError, OSError) as exc:
            why = str(exc) or type(exc).__name__
        finally:
            if self.session is not None:
                self.session.stop()
            try:
                self.sock.close()
            except OSError:
                pass
        self.event("closed", why=why)

    def serve(self) -> None:
        self.sock.sendall(VERSION + b"\r\n")
        theirs = self.version()
        ours = self.kexinit()
        self.send(ours)
        self.exchange(theirs, self.expect(KEXINIT), ours)
        self.login()
        self.connection()

    def version(self) -> bytes:
        """The client's version line, without its line end."""
        for _ in range(20):
            line = self.rfile.readline(256)
            if not line:
                raise EOFError
            if line.startswith(b"SSH-"):
                if not line.startswith((b"SSH-2.0-", b"SSH-1.99-")):
                    raise SSHError("a client of another SSH version")
                return line.rstrip(b"\r\n")
        raise SSHError("no version line")

    @staticmethod
    def kexinit() -> bytes:
        lists = (
            ",".join(KEX),
            KEY_TYPE,
            CIPHER,
            CIPHER,
            MAC,
            MAC,
            "none",
            "none",
            "",
            "",
        )
        return _message(
            KEXINIT, os.urandom(16), *(_string(x) for x in lists), b"\0", _u32(0)
        )

    def exchange(self, theirs: bytes, their_init: bytes, our_init: bytes) -> None:
        """The key exchange (RFC 8731, RFC 5656 4), and the keys it gives each way (RFC 4253 7.2)."""
        r = Reader(their_init[17:])  # past the kind and the cookie
        offered = [r.names() for _ in range(10)]
        chosen = []
        for what, i, ours in (
            ("key exchange", 0, KEX),
            ("host key type", 1, (KEY_TYPE,)),
            ("cipher", 2, (CIPHER,)),
            ("cipher", 3, (CIPHER,)),
            ("compression", 6, ("none",)),
            ("compression", 7, ("none",)),
        ):
            pick = next((a for a in offered[i] if a in ours), None)
            if pick is None:
                self.disconnect(
                    KEY_EXCHANGE_FAILED,
                    f"no {what} in common: the client offers {','.join(offered[i])}",
                )
            chosen.append(pick)
        try:
            seed, public = read_private_key(self.o.host_key)
        except (OSError, ValueError, SSHError) as exc:
            self.disconnect(KEY_EXCHANGE_FAILED, f"no host key: {exc}")
        host_key = key_blob(public)
        self.event("kex", kex=chosen[0], cipher=CIPHER, host_key=fingerprint(host_key))
        q_c = Reader(self.expect(KEX_ECDH_INIT)[1:]).string()
        if len(q_c) != 32:
            self.disconnect(KEY_EXCHANGE_FAILED, "a client key of the wrong size")
        secret = os.urandom(32)
        q_s = x25519(secret, BASE_POINT)
        shared = x25519(secret, q_c)
        if shared == bytes(32):
            self.disconnect(KEY_EXCHANGE_FAILED, "a client key of a small order")
        k = _mpint(int.from_bytes(shared, "big"))
        h = hashlib.sha256(
            b"".join(
                _string(x)
                for x in (theirs, VERSION, their_init, our_init, host_key, q_c, q_s)
            )
            + k
        ).digest()
        self.session_id = self.session_id or h
        signature = _string(KEY_TYPE) + _string(ed25519_sign(seed, h))
        self.send(
            _message(
                KEX_ECDH_REPLY, _string(host_key), _string(q_s), _string(signature)
            )
        )
        self.send(_message(NEWKEYS))
        self.keys_out = self.derive(k, h, b"D")
        self.expect(NEWKEYS)
        self.keys_in = self.derive(k, h, b"C")

    def derive(self, k: bytes, h: bytes, letter: bytes) -> tuple:
        """The cipher's two keys of one way: (the payload's and the MAC's, the length's)."""
        key = hashlib.sha256(k + h + letter + self.session_id).digest()
        key += hashlib.sha256(k + h + key).digest()
        return key[:32], key[32:64]

    def login(self) -> None:
        """ssh-userauth (RFC 4252): the login's public key alone, signed over this session."""
        if Reader(self.expect(SERVICE_REQUEST)[1:]).text() != "ssh-userauth":
            self.disconnect(SERVICE_NOT_AVAILABLE, "only ssh-userauth is served")
        self.send(_message(SERVICE_ACCEPT, _string("ssh-userauth")))
        failure = _message(USERAUTH_FAILURE, _string("publickey"), b"\0")
        for _ in range(20):
            r = Reader(self.expect(USERAUTH_REQUEST)[1:])
            user, service, method = r.text(), r.text(), r.text()
            if method != "publickey" or service != "ssh-connection":
                self.send(failure)
                continue
            signed, kind, blob = r.boolean(), r.text(), r.string()
            try:
                allowed = read_public_key(self.o.authorized_key)
            except (OSError, ValueError, SSHError):
                allowed = None
            ok = user == self.o.user and kind == KEY_TYPE and blob == allowed
            if signed:
                sig = Reader(r.string())
                sig_type, sig_bytes = sig.text(), sig.string()
                data = (
                    _string(self.session_id)
                    + bytes([USERAUTH_REQUEST])
                    + _string(user)
                    + _string(service)
                    + _string("publickey")
                    + b"\1"
                    + _string(kind)
                    + _string(blob)
                )
                public = Reader(blob)
                public.text()
                ok = (
                    ok
                    and sig_type == KEY_TYPE
                    and ed25519_verify(public.string(), data, sig_bytes)
                )
            self.event("auth", user=user, signed=signed, ok=ok)
            if ok and signed:
                self.send(_message(USERAUTH_SUCCESS))
                return
            if ok:
                self.send(_message(USERAUTH_PK_OK, _string(kind), _string(blob)))
            else:
                self.send(failure)
        self.disconnect(NO_MORE_AUTH_METHODS, "too many tries")

    def connection(self) -> None:
        """ssh-connection (RFC 4254): one session, until the client leaves."""
        while True:
            payload = self.receive()
            kind, r = payload[0], Reader(payload[1:])
            if kind in (IGNORE, DEBUG, UNIMPLEMENTED):
                continue
            if kind == DISCONNECT:
                return
            if kind == GLOBAL_REQUEST:
                r.text()
                if r.boolean():
                    self.send(_message(REQUEST_FAILURE))
            elif kind == CHANNEL_OPEN:
                self.open(r)
            elif kind in CHANNEL_MESSAGES:
                if self.session is None or r.u32() != 0:
                    self.disconnect(PROTOCOL_ERROR, "a message for no open channel")
                self.session.take(kind, r)
            elif kind == KEXINIT:
                self.disconnect(
                    KEY_EXCHANGE_FAILED, "a second key exchange: this server makes one"
                )
            else:
                self.send(_message(UNIMPLEMENTED, _u32((self.seq_in - 1) & _M32)))

    def open(self, r: Reader) -> None:
        kind, peer, window, packet = r.text(), r.u32(), r.u32(), r.u32()
        if kind != "session" or self.session is not None:
            self.send(
                _message(
                    CHANNEL_OPEN_FAILURE,
                    _u32(peer),
                    _u32(1),
                    _string("one session per connection here"),
                    _string(""),
                )
            )
            return
        self.session = Session(self, peer, window, packet)
        self.send(
            _message(
                CHANNEL_OPEN_CONFIRMATION,
                _u32(peer),
                _u32(0),
                _u32(WINDOW),
                _u32(MAX_PACKET),
            )
        )


class Session:
    """The session channel, this side's number 0: one command, run with `sh -c`."""

    def __init__(self, conn: Connection, peer: int, window: int, packet: int):
        self.conn, self.peer = conn, peer
        self.window, self.packet = window, max(1, min(packet, MAX_PACKET))
        self.room = (
            threading.Condition()
        )  # the client's window, and whether the channel is closed
        self.closed = False  # this side's CLOSE is sent
        self.proc = None
        # The client's data for the command's input; the size of its extended data, which goes nowhere; None at its end
        self.inbox = queue.Queue()
        self.read = 0  # bytes of the client's data taken since the last WINDOW_ADJUST

    def say(self, kind: int, *parts: bytes) -> bool:
        """A message on the channel, unless it is closed (never one after the close: the lock is the close's too);
        False when it is, or the connection is lost."""
        with self.room:
            if self.closed:
                return False
            try:
                self.conn.send(_message(kind, _u32(self.peer), *parts))
            except OSError:
                return False
            return True

    def take(self, kind: int, r: Reader) -> None:
        """A channel message from the client."""
        if kind == CHANNEL_REQUEST:
            self.request(r)
        elif kind == CHANNEL_DATA:
            self.inbox.put(r.string())
        elif (
            kind == CHANNEL_EXTENDED_DATA
        ):  # nothing for the command: only its room in the window counts
            r.u32()
            self.inbox.put(len(r.string()))
        elif kind == CHANNEL_WINDOW_ADJUST:
            with self.room:
                self.window += r.u32()
                self.room.notify_all()
        elif kind == CHANNEL_EOF:
            self.inbox.put(None)
        elif kind == CHANNEL_CLOSE:
            self.stop()
            self.close()

    def request(self, r: Reader) -> None:
        name, want = r.text(), r.boolean()
        ok = name == "exec" and self.proc is None
        if ok:
            command = r.text()
            self.conn.event("exec", command=command)
            try:
                self.start(command)
            except OSError as exc:
                self.conn.event("exit", status=None, why=str(exc))
                ok = False
        if want:
            self.say(CHANNEL_SUCCESS if ok else CHANNEL_FAILURE)
        if ok:
            pumps = [
                threading.Thread(target=self.pump, args=(self.proc.stdout, None)),
                threading.Thread(target=self.pump, args=(self.proc.stderr, 1)),
            ]
            for t in (*pumps, threading.Thread(target=self.feed)):
                t.daemon = True
                t.start()
            threading.Thread(target=self.finish, args=(pumps,), daemon=True).start()
        elif name == "exec" and self.proc is None:
            self.close()

    def start(self, command: str) -> None:
        o = self.conn.o
        env = {
            "HOME": o.home,
            "USER": o.user,
            "LOGNAME": o.user,
            "PATH": o.path,
            "SHELL": "/bin/sh",
            "LANG": "C.UTF-8",
        }
        self.proc = subprocess.Popen(
            ["/bin/sh", "-c", command],
            cwd=o.home,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )

    def adjust(self, n: int) -> None:
        """N more bytes of the client's taken: once half the window is, the client may send that much again."""
        self.read += n
        if self.read >= WINDOW // 2:
            n, self.read = self.read, 0
            self.say(CHANNEL_WINDOW_ADJUST, _u32(n))

    def feed(self) -> None:
        """The client's data into the command's input, then its end."""
        stdin = self.proc.stdin
        while True:
            data = self.inbox.get()
            if data is None:
                break
            if isinstance(data, int):  # extended data's size: nothing to write
                self.adjust(data)
                continue
            try:
                stdin.write(data)
                stdin.flush()
            except (
                OSError,
                ValueError,
            ):  # the command closed its input: the rest goes nowhere
                pass
            self.adjust(len(data))
        try:
            stdin.close()
        except OSError:
            pass

    def pump(self, stream, ext) -> None:
        """The command's output (EXT None) or its errors (EXT 1) to the client, as its window has room."""
        fd = stream.fileno()
        while True:
            try:
                data = os.read(fd, MAX_PACKET)
            except OSError:
                data = b""
            if not data:
                break
            while data:
                with self.room:
                    while self.window <= 0 and not self.closed:
                        self.room.wait(1)
                    if self.closed:
                        return
                    n = min(len(data), self.window, self.packet)
                    self.window -= n
                chunk, data = data[:n], data[n:]
                if ext is None:
                    self.say(CHANNEL_DATA, _string(chunk))
                else:
                    self.say(CHANNEL_EXTENDED_DATA, _u32(ext), _string(chunk))

    def finish(self, pumps: list) -> None:
        """Once the command has ended and all it wrote is sent: its exit status, the end of its data, the close."""
        for t in pumps:
            t.join()
        status = self.proc.wait()
        self.conn.event("exit", status=status)
        if status >= 0:
            self.say(CHANNEL_REQUEST, _string("exit-status"), b"\0", _u32(status))
        else:
            name = signal.Signals(-status).name[3:]
            self.say(
                CHANNEL_REQUEST,
                _string("exit-signal"),
                b"\0",
                _string(name),
                b"\0",
                _string(""),
                _string(""),
            )
        self.say(CHANNEL_EOF)
        self.close()

    def close(self) -> None:
        with self.room:
            if self.closed:
                return
            sent = self.say(CHANNEL_CLOSE)
            self.closed = True
            self.room.notify_all()
        if not sent:
            self.stop()

    def stop(self) -> None:
        """The command and what it started in its session stopped, while they run; its input ended."""
        self.inbox.put(None)
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except OSError:
                pass


class Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host-key", required=True)
    ap.add_argument("--authorized-key", required=True)
    ap.add_argument("--user", required=True)
    ap.add_argument("--home", required=True)
    ap.add_argument("--address", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=22)
    ap.add_argument("--path", default="/usr/local/bin:/usr/bin:/bin")
    ap.add_argument("--log", default="")
    ap.add_argument("--ready", default="")
    o = ap.parse_args(argv)
    lock = threading.Lock()
    numbers = itertools.count(1)

    def log(event: str, **fields) -> None:
        if o.log:
            line = json.dumps(dict(t=round(time.time(), 3), event=event, **fields))
            with lock, open(o.log, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            self.request.settimeout(IDLE_S)
            with lock:
                n = next(numbers)
            peer = self.client_address[0]

            def event(what: str, **fields) -> None:
                log(what, conn=n, peer=peer, **fields)

            event("connect")
            Connection(self.request, o, event).run()

    Server.address_family = socket.AF_INET6 if ":" in o.address else socket.AF_INET
    with Server((o.address, o.port), Handler) as srv:
        port = srv.server_address[1]
        log("listening", port=port)
        if o.ready:
            pathlib.Path(o.ready).write_text(f"{port}\n", encoding="ascii")
        srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
