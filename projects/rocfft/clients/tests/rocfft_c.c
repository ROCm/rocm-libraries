/* rocfft_c.c
 *
 * Verifies that rocfft/rocfft.h is self-sufficient when compiled as plain C.
 * rocfft/rocfft.h must be the very first include so that any missing
 * dependency (e.g. <stddef.h> for size_t) would surface as a compile error.
 */

#include <rocfft/rocfft.h>

int main(void)
{
    return 0;
}
