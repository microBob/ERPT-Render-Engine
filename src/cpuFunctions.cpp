//
// Created by microbobu on 12/19/20.
//

#include "../include/kernels.cuh"

int detectFloatPrecision() {
	float num = 1.0f / 3.0f;
	unsigned int convert = (int) (num * pow(10, 9));

	unsigned int lastThree[]{(convert % 1000 - convert % 100) / 100, (convert % 100 - convert % 10) / 10,
	                         convert % 10};
	for (int i = 0; i < 3; ++i) {
		if (lastThree[i] != 3) {
			return 6 + i;
		}
	}

	return 9;
}
