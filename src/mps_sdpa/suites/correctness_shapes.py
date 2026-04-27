"""Correctness shape matrix — broader than ranking suite. Causal + edges + non-contig."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from typing import Iterator


@dataclass(frozen=True)
class Case:
    case_id: str
    B: int
    H: int
    Lq: int
    Lkv: int
    D: int
    dtype: str
    mask: str
    contiguous: bool
    dropout_p: float


def iter_cases() -> Iterator[Case]:
    for B, H, Lq, Lkv, D in product(
        [1, 2, 4], [4, 8, 16], [32, 128, 512, 2048], [32, 128, 512, 2048], [32, 64, 128]
    ):
        if Lq != Lkv and (Lq, Lkv) not in [(128, 512), (512, 128), (512, 2048), (2048, 512)]:
            continue
        for dtype in ["fp16", "bf16", "fp32"]:
            for mask in ["none", "bool_b1lk", "causal"]:
                yield Case(
                    case_id=f"c_B{B}H{H}Lq{Lq}Lkv{Lkv}D{D}_{dtype}_{mask}",
                    B=B, H=H, Lq=Lq, Lkv=Lkv, D=D,
                    dtype=dtype, mask=mask, contiguous=True, dropout_p=0.0,
                )

    for mask in ["bool_bhlk", "additive_float", "empty_row"]:
        for dtype in ["fp16", "fp32"]:
            yield Case(
                case_id=f"c_edge_mask_{mask}_{dtype}",
                B=1, H=4, Lq=128, Lkv=128, D=64,
                dtype=dtype, mask=mask, contiguous=True, dropout_p=0.0,
            )

    for dtype in ["fp16", "fp32"]:
        yield Case(
            case_id=f"c_noncontig_{dtype}",
            B=1, H=4, Lq=128, Lkv=128, D=64,
            dtype=dtype, mask="none", contiguous=False, dropout_p=0.0,
        )

    yield Case(
        case_id="c_dropout_fp16",
        B=1, H=4, Lq=128, Lkv=128, D=64,
        dtype="fp16", mask="none", contiguous=True, dropout_p=0.1,
    )


def iter_extended_cases() -> Iterator[Case]:
    """Broader D/H/B + non-power-of-2 Lq/Lkv coverage.

    Surfaces any hard failures across realistic shape/dtype/mask combinations
    so they can either be fixed or get explicit fallback handling. All cases
    are bf16 to keep runtime manageable.
    """
    # D sweep (fixed B=1,H=8,Lq=Lkv=512). D=256 matters for wide-Q-head models.
    for D in (32, 64, 96, 128, 192, 256):
        yield Case(
            case_id=f"c_ext_D{D}_bf16",
            B=1, H=8, Lq=512, Lkv=512, D=D,
            dtype="bf16", mask="none", contiguous=True, dropout_p=0.0,
        )
    # H sweep (fixed B=1,Lq=Lkv=512,D=64).
    for H in (1, 2, 4, 8, 16, 32):
        yield Case(
            case_id=f"c_ext_H{H}_bf16",
            B=1, H=H, Lq=512, Lkv=512, D=64,
            dtype="bf16", mask="none", contiguous=True, dropout_p=0.0,
        )
    # B sweep (fixed H=8,Lq=Lkv=512,D=64). B=32 stresses the B-major graph layout.
    for B in (1, 2, 4, 8, 32):
        yield Case(
            case_id=f"c_ext_B{B}_bf16",
            B=B, H=8, Lq=512, Lkv=512, D=64,
            dtype="bf16", mask="none", contiguous=True, dropout_p=0.0,
        )
    # Non-power-of-2 Lq/Lkv (self-attn shape: Lq==Lkv).
    for L in (777, 1345, 3141):
        yield Case(
            case_id=f"c_ext_L{L}_bf16",
            B=1, H=8, Lq=L, Lkv=L, D=64,
            dtype="bf16", mask="none", contiguous=True, dropout_p=0.0,
        )
    # Non-POT cross-attn: Lq != Lkv
    for Lq, Lkv in [(777, 1345), (1345, 777), (3141, 777)]:
        yield Case(
            case_id=f"c_ext_Lq{Lq}_Lkv{Lkv}_bf16",
            B=1, H=8, Lq=Lq, Lkv=Lkv, D=64,
            dtype="bf16", mask="none", contiguous=True, dropout_p=0.0,
        )
