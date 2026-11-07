// Linux/x86-64 CPU-only adapter for Clang's native OpenCL C translation.
// Groups run serially. A work-item yields at each local-memory barrier.
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ucontext.h>
#include <vector>

namespace {
using Unary = void (*)(void *, const void *);
using Product = void (*)(void *, const void *, const void *);
ucontext_t caller;
std::vector<ucontext_t> contexts;
std::vector<size_t> groupOffsets;
size_t active, firstId, endId, groupSize;
Unary transform;
Product product;
void *data;
const void *multiplicand, *roots;
}

size_t get_global_id(unsigned) { return groupOffsets[active] + active; }
uint32_t mul_hi(uint32_t a, uint32_t b) { return uint32_t((uint64_t(a) * b) >> 32); }
extern "C" unsigned long cl_abs(long) asm("_Z3absl");
unsigned long cl_abs(long v) { return v < 0 ? 0ul - static_cast<unsigned long>(v) : static_cast<unsigned long>(v); }

void barrier(unsigned)
{
	const size_t previous = active;
	active = (active + 1) % groupSize;
	if (swapcontext(&contexts[previous], &contexts[active]) != 0) std::abort();
}

static void runLane(int index)
{
	const auto lane = static_cast<size_t>(index);
	for (size_t offset = firstId; offset < endId; offset += groupSize)
	{
		groupOffsets[lane] = offset;
		if (product) product(data, multiplicand, roots);
		else transform(data, roots);
		barrier(0); // Finish all consumers before reusing this kernel's local array.
	}
	active = (lane + 1) % groupSize;
}

extern "C" void run_groups(Unary fn, Product mul, void *z, const void *zp, const void *w,
	size_t start, size_t count, size_t local)
{
	if (local < 2 || count % local || start % local || (fn == nullptr) == (mul == nullptr)) std::abort();
	transform = fn; product = mul; data = z; multiplicand = zp; roots = w;
	groupSize = local; firstId = start; endId = start + count; active = 0;
	contexts.resize(local);
	groupOffsets.resize(local);
	const size_t stackBytes = 32 * 1024;
	std::vector<unsigned char> stacks(local * stackBytes);
	for (size_t i = 0; i < local; ++i)
	{
		auto &ctx = contexts[i];
		if (getcontext(&ctx) != 0) std::abort();
		ctx.uc_stack.ss_sp = stacks.data() + i * stackBytes;
		ctx.uc_stack.ss_size = stackBytes;
		ctx.uc_link = i + 1 == local ? &caller : &contexts[i + 1];
		makecontext(&ctx, reinterpret_cast<void (*)()>(runLane), 1, static_cast<int>(i));
	}
	if (swapcontext(&caller, &contexts[0]) != 0) std::abort();
}
