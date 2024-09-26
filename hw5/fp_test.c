#include <stdio.h>

int main() {
    size_t N = 32 * 1024 * 1024;
    float sum = 0;
    for (int i = 0; i < N; i++) {
        sum += 1.0;
    }
    float test = 32 * 1024 * 1024;
    printf("test \t= %f\n", test);
    printf("sum \t= %f\n", sum);
    printf("N \t= %lu\n", N);
}