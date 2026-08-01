from typing import Optional

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="The text prompt to continue")
    max_new_tokens: int = Field(30, ge=1, le=200)
    lookahead: int = Field(4, ge=1, le=8, description="Draft tokens proposed per round")
    temperature: float = Field(1.0, gt=0, le=2.0)
    seed: int = Field(0)
    compare_to_naive: bool = Field(
        False, description="Also run plain (non-speculative) decoding and include a speedup comparison"
    )


class GenerateResponse(BaseModel):
    text: str
    elapsed_seconds: float
    tokens_per_second: float
    target_forward_passes: int
    acceptance_rate: float
    device: str
    naive_elapsed_seconds: Optional[float] = None
    naive_text: Optional[str] = None
    speedup: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    device: str
    draft_model: str
    target_model: str
