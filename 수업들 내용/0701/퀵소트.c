#include <stdio.h>


int a[1000];  // 또는 원하는 크기

void quickSort(int start, int end) {
    if (start >= end) return;
    int key = start, i = start + 1, j = end, temp;
    while (i <= j) {
		while (i <= end && a[i] <= a[key]) i++; // i는 key보다 큰 값을 찾기 위해 증가
		while (j > start && a[j] >= a[key]) j--; // j는 key보다 작은 값을 찾기 위해 감소

        if (i > j) swap(&a[key], &a[j]);
        else swap(&a[i], &a[j]);
    }
	// 재귀 호출
    quickSort(start, j - 1);
    quickSort(j + 1, end);
}

void swap(int* x, int* y) {
    int temp = *x;
    *x = *y;
    *y = temp;
}

int main(void) {

    int n;
    scanf("%d", &n);
    for (int i = 0; i < n; i++) {
        scanf("%d", &a[i]);
    }
    // 퀵소트
    quickSort(0, n - 1);

    for (int i = 0; i < n; i++) {
        printf("%d ", a[i]);
    }
}
