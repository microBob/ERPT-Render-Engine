#pragma clang diagnostic push
#pragma ide diagnostic ignored "bugprone-reserved-identifier"
//
// Created by microbobu on 2/21/21.
//
#include "../include/optixLaunchParameters.h"
#include "optix_device.h"

/// Launch Parameters
extern "C" __constant__ OptixLaunchParameters optixLaunchParameters;

enum {
	SURFACE_RAY_TYPE = 0,
	RAY_TYPE_COUNT
};

/// Utility functions
__device__ float3 normalizeVectorGPU(float3 vector) {
	const auto r_normal = rnorm3df(vector.x, vector.y, vector.z);

	return make_float3(vector.x * r_normal, vector.y * r_normal, vector.z * r_normal);
}

__device__ float3 vectorCrossProductGPU(float3 vectorA, float3 vectorB) {
	return make_float3(vectorA.y * vectorB.z - vectorA.z * vectorB.y, vectorA.z * vectorB.x - vectorA.x * vectorB.z,
	                   vectorA.x * vectorB.y - vectorA.y * vectorB.x);
}

__device__ float vectorDotProductGPU(float3 vectorA, float3 vectorB) {
	return vectorA.x * vectorB.x + vectorA.y * vectorB.y + vectorA.z * vectorB.z;
}

/// Payload management
static __forceinline__ __device__ void *unpackPointer(uint32_t i0, uint32_t i1) {
	const uint64_t rawPointer = static_cast<uint64_t>(i0) << 32 | i1;
	void *pointer = reinterpret_cast<void *>(rawPointer);
	return pointer;
}

static __forceinline__ __device__ void packPointer(void *pointer, uint32_t &i0, uint32_t &i1) {
	const auto rawPointer = reinterpret_cast<uint64_t>(pointer);
	i0 = rawPointer >> 32;
	i1 = rawPointer & 0x00000000ffffffff;
}

template<typename T>
static __forceinline__ __device__ T *getPerRayData() {
	const uint32_t u0 = optixGetPayload_0();
	const uint32_t u1 = optixGetPayload_1();
	return reinterpret_cast<T *>( unpackPointer(u0, u1));
}

/// Ray generation program
__device__ float mutatedMutationNumber(unsigned int mutationNumberIndex, unsigned int indexShift) {
	float jumpSize;
	if (optixLaunchParameters.samples.index % 2 == 0) {
		jumpSize = 0.5f;
	} else {
		jumpSize = 0.005f;
	}

	float newMutationNumberToUse;
	if (optixLaunchParameters.newMutationNumbers[mutationNumberIndex + indexShift] ==
	    optixLaunchParameters.newMutationNumbers[mutationNumberIndex + indexShift]) {
		if (indexShift == optixLaunchParameters.traceDepth * 2) {
			newMutationNumberToUse = optixLaunchParameters.newMutationNumbers[mutationNumberIndex];
		} else {
			newMutationNumberToUse = optixLaunchParameters.newMutationNumbers[mutationNumberIndex + indexShift + 1];
		}
	}

	float randomNumber = 1.0f - 2.0f * newMutationNumberToUse;
	float randomJump = jumpSize * randomNumber;

	float mutatedNumber = optixLaunchParameters.curMutationNumbers[mutationNumberIndex + indexShift] + randomJump;

	if (mutatedNumber > 1.0f) {
		mutatedNumber -= 1.0f;
	} else if (mutatedNumber < 0.0f) {
		mutatedNumber += 1.0f;
	}

//	if (mutatedNumber == 0.5f) {
//		printf("> %f, %f\n", randomNumber, randomJump);
//	}

	return mutatedNumber;


//	return fminf(fabsf(optixLaunchParameters.curMutationNumbers[mutationNumberIndex + indexShift] + randomJump));

//	return fminf(fabsf(optixLaunchParameters.curMutationNumbers[mutationNumberIndex + indexShift] +
//	                   jumpSize *
//	                   (1.0f - 2 * optixLaunchParameters.newMutationNumbers[mutationNumberIndex + indexShift])),
//	             1.0f);
}

extern "C" __global__ void __raygen__renderFrame() {
	// Get index and camera
	const unsigned int ix = optixGetLaunchIndex().x;
	const unsigned int iy = optixGetLaunchIndex().y;
	const unsigned int mutationNumberIndex = ix + iy * optixLaunchParameters.frame.frameBufferSize.x;
	unsigned int screenX = llrintf(static_cast<float>(optixLaunchParameters.frame.frameBufferSize.x - 1) *
	                               optixLaunchParameters.curMutationNumbers[mutationNumberIndex]);
	unsigned int screenY = llrintf(static_cast<float>(optixLaunchParameters.frame.frameBufferSize.y - 1) *
	                               optixLaunchParameters.curMutationNumbers[mutationNumberIndex + 1]);
	unsigned int pixelIndex = screenX + screenY * optixLaunchParameters.frame.frameBufferSize.x;

	const auto &camera = optixLaunchParameters.camera;

//	if (mutationNumberIndex == 585) {
//		printf("Access: ");
//		for (int i = 0; i < optixLaunchParameters.traceDepth + 2; ++i) {
//			printf("%f, ", optixLaunchParameters.curMutationNumbers[mutationNumberIndex + i]);
//		}
//		printf("\n");
//	}

	/// Starting ray from camera
	// Create per ray data
	PerRayData rayData;
	uint32_t payload0, payload1;
	packPointer(&rayData, payload0, payload1);

	// Create base screen ray
//	auto screen = make_float2(
//		(static_cast<float>(screenX) + 0.5f) /
//		static_cast<float>(optixLaunchParameters.frame.frameBufferSize.x),
//		(static_cast<float>(screenY) + 0.5f) /
//		static_cast<float>(optixLaunchParameters.frame.frameBufferSize.y));
	auto screen = make_float2(optixLaunchParameters.curMutationNumbers[mutationNumberIndex],
	                          optixLaunchParameters.curMutationNumbers[mutationNumberIndex + 1]);
	auto screenMinus = make_float2(screen.x - 0.5f, screen.y - 0.5f);
	auto horizontalTimesScreenMinus = make_float3(screenMinus.x * camera.horizontal.x,
	                                              screenMinus.x * camera.horizontal.y,
	                                              screenMinus.x * camera.horizontal.z);
	auto verticalTimesScreenMinus = make_float3(screenMinus.y * camera.vertical.x,
	                                            screenMinus.y * camera.vertical.y,
	                                            screenMinus.y * camera.vertical.z);
	auto rawRayDirection = make_float3(
		camera.direction.x + horizontalTimesScreenMinus.x + verticalTimesScreenMinus.x,
		camera.direction.y + horizontalTimesScreenMinus.y + verticalTimesScreenMinus.y,
		camera.direction.z + horizontalTimesScreenMinus.z + verticalTimesScreenMinus.z);

	float3 rayOrigin = camera.position;
	float3 rayDirectionNormalized = normalizeVectorGPU(rawRayDirection);

	// Trace
	optixTrace(optixLaunchParameters.optixTraversableHandle,
	           rayOrigin,
	           rayDirectionNormalized,
	           0.001f, // Needs to have gone somewhere
	           1e20f,
	           0.0f,
	           OptixVisibilityMask(255),
	           OPTIX_RAY_FLAG_DISABLE_ANYHIT,
	           SURFACE_RAY_TYPE,
	           RAY_TYPE_COUNT,
	           SURFACE_RAY_TYPE,
	           payload0,
	           payload1);

	colorVector baseColor;
	bool firstRaySuccessful = false;
	// Complete first ray tracing
	if (rayData.normal.x + rayData.normal.y + rayData.normal.z != 0) {
		baseColor = rayData.color;
		int depthIndex;

		// Increment Energy at pixel if a light source was hit
		if (rayData.light) {
			firstRaySuccessful = true;
		} else { // Else, continue with second ray
			/// Reflected ray
			for (depthIndex = 0; depthIndex < optixLaunchParameters.traceDepth; ++depthIndex) {
				// Create ray
				const float r = sqrt(
					optixLaunchParameters.curMutationNumbers[mutationNumberIndex + 2 + depthIndex * 2]);
				const float phi = 2 * 3.1415f *
				                  optixLaunchParameters.curMutationNumbers[mutationNumberIndex + 2 + depthIndex * 2 +
				                                                           1];
				const float circleX = r * cos(phi);
				const float circleY = r * sin(phi);
				const float circleZ = sqrt(1 - (r * r));
				const float3 newDirection = make_float3(
					rayData.xAxis.x * circleX + rayData.yAxis.x * circleY + rayData.normal.x * circleZ,
					rayData.xAxis.y * circleX + rayData.yAxis.y * circleY + rayData.normal.y * circleZ,
					rayData.xAxis.z * circleX + rayData.yAxis.z * circleY + rayData.normal.z * circleZ);

				rayOrigin = rayData.location;
				rayDirectionNormalized = newDirection;


				// Trace
				optixTrace(optixLaunchParameters.optixTraversableHandle,
				           rayOrigin,
				           rayDirectionNormalized,
				           0.01f, // Needs to have gone somewhere
				           1e20f,
				           0.0f,
				           OptixVisibilityMask(255),
				           OPTIX_RAY_FLAG_DISABLE_ANYHIT,
				           SURFACE_RAY_TYPE,
				           RAY_TYPE_COUNT,
				           SURFACE_RAY_TYPE,
				           payload0,
				           payload1);

				// Stop if there was a miss
				if (rayData.normal.x + rayData.normal.y + rayData.normal.z == 0) {
					break;
				}
				// If there's light, increment data
				if (rayData.light) {
					firstRaySuccessful = true;
					break;
				}
			}
		}
	}
//
//	if (firstRaySuccessful) {
//		printf("First ray successful\n");
//	}

	// Make proposal ray
	const float2 proposedScreenXY = make_float2(mutatedMutationNumber(mutationNumberIndex, 0),
	                                            mutatedMutationNumber(mutationNumberIndex, 1));
	const auto proposedScreenX = llrintf(
		static_cast<float>(optixLaunchParameters.frame.frameBufferSize.x - 1) * proposedScreenXY.x);
	const auto proposedScreenY = llrintf(
		static_cast<float>(optixLaunchParameters.frame.frameBufferSize.y - 1) * proposedScreenXY.y);
	const auto proposedPixelIndex = proposedScreenX + proposedScreenY * optixLaunchParameters.frame.frameBufferSize.x;

//	printf("%f, %f\n", optixLaunchParameters.curMutationNumbers[mutationNumberIndex],
//	       mutatedMutationNumber(mutationNumberIndex, 0));

//	screen = make_float2(
//		(static_cast<float>(proposedScreenX) + 0.5f) /
//		static_cast<float>(optixLaunchParameters.frame.frameBufferSize.x),
//		(static_cast<float>(proposedScreenY) + 0.5f) /
//		static_cast<float>(optixLaunchParameters.frame.frameBufferSize.y));
	screenMinus = make_float2(proposedScreenXY.x - 0.5f, proposedScreenXY.y - 0.5f);
	horizontalTimesScreenMinus = make_float3(screenMinus.x * camera.horizontal.x,
	                                         screenMinus.x * camera.horizontal.y,
	                                         screenMinus.x * camera.horizontal.z);
	verticalTimesScreenMinus = make_float3(screenMinus.y * camera.vertical.x,
	                                       screenMinus.y * camera.vertical.y,
	                                       screenMinus.y * camera.vertical.z);
	rawRayDirection = make_float3(
		camera.direction.x + horizontalTimesScreenMinus.x + verticalTimesScreenMinus.x,
		camera.direction.y + horizontalTimesScreenMinus.y + verticalTimesScreenMinus.y,
		camera.direction.z + horizontalTimesScreenMinus.z + verticalTimesScreenMinus.z);

	rayOrigin = camera.position;
	rayDirectionNormalized = normalizeVectorGPU(rawRayDirection);

	// Trace
	optixTrace(optixLaunchParameters.optixTraversableHandle,
	           rayOrigin,
	           rayDirectionNormalized,
	           0.01f, // Needs to have gone somewhere
	           1e20f,
	           0.0f,
	           OptixVisibilityMask(255),
	           OPTIX_RAY_FLAG_DISABLE_ANYHIT,
	           SURFACE_RAY_TYPE,
	           RAY_TYPE_COUNT,
	           SURFACE_RAY_TYPE,
	           payload0,
	           payload1);

	colorVector secondBaseColor;
	bool proposedRaySuccessful = false;
	// Complete proposed ray tracing
	if (rayData.normal.x + rayData.normal.y + rayData.normal.z != 0) {
		secondBaseColor = rayData.color; // Set base color

		if (rayData.light) {
			proposedRaySuccessful = true;
		} else { // Else, continue with bounce
			/// Reflected ray
			for (int depthIndex = 0; depthIndex < optixLaunchParameters.traceDepth; ++depthIndex) {
				// Create ray
				const float r = sqrt(mutatedMutationNumber(mutationNumberIndex + 2, depthIndex * 2));
				const float phi =
					2 * 3.1415f * mutatedMutationNumber(mutationNumberIndex + 2, depthIndex * 2 + 1);
				const float circleX = r * cos(phi);
				const float circleY = r * sin(phi);
				const float circleZ = sqrt(1 - (r * r));
				const float3 newDirection = make_float3(
					rayData.xAxis.x * circleX + rayData.yAxis.x * circleY + rayData.normal.x * circleZ,
					rayData.xAxis.y * circleX + rayData.yAxis.y * circleY + rayData.normal.y * circleZ,
					rayData.xAxis.z * circleX + rayData.yAxis.z * circleY + rayData.normal.z * circleZ);

				rayOrigin = rayData.location;
				rayDirectionNormalized = newDirection;


				// Trace
				optixTrace(optixLaunchParameters.optixTraversableHandle,
				           rayOrigin,
				           rayDirectionNormalized,
				           0.001f, // Needs to have gone somewhere
				           1e20f,
				           0.0f,
				           OptixVisibilityMask(255),
				           OPTIX_RAY_FLAG_DISABLE_ANYHIT,
				           SURFACE_RAY_TYPE,
				           RAY_TYPE_COUNT,
				           SURFACE_RAY_TYPE,
				           payload0,
				           payload1);

				// Stop if there was a miss
				if (rayData.normal.x + rayData.normal.y + rayData.normal.z == 0) {
					break;
				}
				// Indicated there's a light and update mutation numbers
				if (rayData.light) {
					proposedRaySuccessful = true;

					// Copy used mutation numbers into curMutations
					optixLaunchParameters.curMutationNumbers[mutationNumberIndex] = proposedScreenXY.x;
					optixLaunchParameters.curMutationNumbers[mutationNumberIndex + 1] = proposedScreenXY.y;
					for (int i = 0; i < depthIndex; ++i) {
						optixLaunchParameters.curMutationNumbers[mutationNumberIndex + 2 +
						                                         i * 2] = mutatedMutationNumber(
							mutationNumberIndex + 2, i * 2);
						optixLaunchParameters.curMutationNumbers[mutationNumberIndex + 2 + i * 2 +
						                                         1] = mutatedMutationNumber(
							mutationNumberIndex + 2, i * 2 + 1);
					}
//						if (mutationNumberIndex == 585) {
//							if (mutationNumberIndex == 585) {
//								printf("Copy: ");
//								for (int i = 0; i < optixLaunchParameters.traceDepth + 2; ++i) {
//									printf("%f, ", optixLaunchParameters.curMutationNumbers[mutationNumberIndex + i]);
//								}
//								printf("\n");
//							}
//						}
					break;
				}
			}
		}
	}

	// Update color at that pixel
	if (proposedRaySuccessful) {
		const float baseColorValueSum =
			(secondBaseColor.r + secondBaseColor.g + secondBaseColor.b) / rayData.energy *
			static_cast<float>(optixLaunchParameters.samples.total);

//			if (pixelIndex == proposedPixelIndex) {
//				printf("Mutated Same\n");
//			}
//			printf("%u vs %u\n", pixelIndex, proposedPixelIndex);

		atomicAdd(&optixLaunchParameters.frame.frameColorBuffer[proposedPixelIndex].r,
		          secondBaseColor.r / baseColorValueSum);
		atomicAdd(&optixLaunchParameters.frame.frameColorBuffer[proposedPixelIndex].g,
		          secondBaseColor.g / baseColorValueSum);
		atomicAdd(&optixLaunchParameters.frame.frameColorBuffer[proposedPixelIndex].b,
		          secondBaseColor.b / baseColorValueSum);
	} else if (firstRaySuccessful) {
		const float baseColorValueSum =
			(baseColor.r + baseColor.g + baseColor.b) / rayData.energy *
			static_cast<float>(optixLaunchParameters.samples.total);
		atomicAdd(&optixLaunchParameters.frame.frameColorBuffer[pixelIndex].r, baseColor.r / baseColorValueSum);
		atomicAdd(&optixLaunchParameters.frame.frameColorBuffer[pixelIndex].g, baseColor.g / baseColorValueSum);
		atomicAdd(&optixLaunchParameters.frame.frameColorBuffer[pixelIndex].b, baseColor.b / baseColorValueSum);
	} else {
		// Get new values for base
		for (int i = 0; i < optixLaunchParameters.traceDepth + 2; ++i) {
			optixLaunchParameters.curMutationNumbers[mutationNumberIndex +
			                                         i] = optixLaunchParameters.newMutationNumbers[mutationNumberIndex +
			                                                                                       i];
		}
	}
}

/// Miss program
extern "C" __global__ void __miss__radiance() {
	PerRayData &perRayData = *(PerRayData *) getPerRayData<PerRayData>();
	const auto zeroVector = make_float3(0, 0, 0);
	perRayData = {zeroVector, zeroVector, zeroVector, zeroVector, {}, 0, false};
}

/// Hit program
extern "C" __global__ void __closesthit__radiance() {
	const TriangleMeshSBTData &sbtData = *(const TriangleMeshSBTData *) optixGetSbtDataPointer();
	PerRayData &perRayData = *(PerRayData *) getPerRayData<PerRayData>();

	// Return early if this was a light source
	if (sbtData.kind == Light) {
		perRayData = {make_float3(0, 0, 0), make_float3(1, 0, 0), make_float3(0, 0, 0), make_float3(0, 0, 0),
		              sbtData.color, sbtData.energy, true};
		return;
	}

	// Essential hit data
	const float3 rayDir = optixGetWorldRayDirection();
	const float3 rayOrigin = optixGetWorldRayOrigin();
	const float rayLength = optixGetRayTmax();

	// Check if valid hit
	if (rayLength <= 0.0001) { // Basically 0
		printf("No ray length, skipping\n");
		return;
	}

	const float3 hitLocation = make_float3(rayOrigin.x + rayLength * rayDir.x, rayOrigin.y + rayLength * rayDir.y,
	                                       rayOrigin.z + rayLength * rayDir.z);

	// Surface normal
	const int primitiveIndex = optixGetPrimitiveIndex();
	const int3 index = sbtData.index[primitiveIndex];
	const float3 &vertexA = sbtData.vertex[index.x];
	const float3 &vertexB = sbtData.vertex[index.y];
	const float3 &vertexC = sbtData.vertex[index.z];
	const auto vertexBMinusA = make_float3(vertexB.x - vertexA.x, vertexB.y - vertexA.y, vertexB.z - vertexA.z);
	const auto vertexCMinusA = make_float3(vertexC.x - vertexA.x, vertexC.y - vertexA.y, vertexC.z - vertexA.z);
	const float3 normalAxis = normalizeVectorGPU(vectorCrossProductGPU(vertexBMinusA, vertexCMinusA));
	const colorVector normalColor = {(normalAxis.x + 1) / 2, (normalAxis.y + 1) / 2, (normalAxis.z + 1) / 2};

	// Second Axis
	// TODO: causing bad indirect lighting
	const unsigned int ix = optixGetLaunchIndex().x;
	const unsigned int iy = optixGetLaunchIndex().y;
	const unsigned int mutationNumberIndex = ix + iy * optixLaunchParameters.frame.frameBufferSize.x;
	const float3 yAxis = normalizeVectorGPU(
		vectorCrossProductGPU(make_float3(optixLaunchParameters.curMutationNumbers[mutationNumberIndex],
		                                  optixLaunchParameters.curMutationNumbers[mutationNumberIndex + 1],
		                                  optixLaunchParameters.curMutationNumbers[mutationNumberIndex + 2]),
		                      normalAxis));
//	const float3 yAxis = normalizeVectorGPU(vectorCrossProductGPU(optixLaunchParameters.camera.direction, normalAxis));

	// Third Axis
	const float3 xAxis = normalizeVectorGPU(vectorCrossProductGPU(normalAxis, yAxis));

	// Encode per ray data
	perRayData = {hitLocation, normalAxis, xAxis, yAxis, sbtData.color, sbtData.energy, sbtData.kind == Light};
}
extern "C" __global__ void __anyhit__radiance() {}

#pragma clang diagnostic pop