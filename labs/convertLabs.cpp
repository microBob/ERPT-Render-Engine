//
// Created by microbobu on 12/19/20.
//
#include <iostream>
#include <sstream>
#include <cmath>
#include <chrono>

using namespace std;
using namespace chrono;

int main() {
	size_t pixDataSize = 1920 * 1080 * 4 * sizeof(float);
	auto pixData = (float *) malloc(pixDataSize);
	pixData[0] = 102.0f / 255.0f;
	pixData[1] = 153.0f / 255.0f;
	pixData[2] = 255.0f / 255.0f;
	pixData[3] = 1.0f;

	// figure out float precision
	unsigned int targetPrecision = 9;
	float num = 1.0f / 3.0f;
	unsigned int convert = (int) (num * pow(10, 9));
	cout << convert << endl;

	unsigned int lastThree[]{(convert % 1000 - convert % 100) / 100, (convert % 100 - convert % 10) / 10,
	                         convert % 10};
	for (int i = 0; i < 3; ++i) {
		if (lastThree[i] != 3) {
			targetPrecision = 6 + i;
			break;
		}
	}
	cout << "Using float precision: " << targetPrecision << endl << endl;

	auto beforeConvert = high_resolution_clock::now();

	stringstream ss;
	ss << targetPrecision << ' ';
	for (int i = 0; i < pixDataSize / sizeof(float); ++i) {
		ss << (int) (pixData[i] * pow(10, targetPrecision)) << ' ';
	}

	auto afterConvert = high_resolution_clock::now();
	int convertDur = duration_cast<milliseconds>(afterConvert - beforeConvert).count();
	cout << "Convert took: " << convertDur << " ms" << endl;


	free(pixData);
	return 0;
}
