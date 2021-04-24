//
// Created by microbobu on 3/14/21.
//

#ifndef ERPT_RENDER_ENGINE_TYPES_H
#define ERPT_RENDER_ENGINE_TYPES_H

struct colorVector {
	// Default to a black pixel with full alpha
	float r = 0;
	float g = 0;
	float b = 0;
	float a = 1;
};

enum MeshKind {
	Mesh,
	Light
};

#endif //ERPT_RENDER_ENGINE_TYPES_H
