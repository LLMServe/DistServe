import triton
import triton.language as tl

@triton.jit
def f(a: tl.constexpr, b):
	pass
