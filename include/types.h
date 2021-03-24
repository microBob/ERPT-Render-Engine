//
// Created by microbobu on 3/14/21.
//

#ifndef ERPT_RENDER_ENGINE_TYPES_H
#define ERPT_RENDER_ENGINE_TYPES_H

struct vector2i {
	int x, y;
};
struct vector2f {
	float x, y;
};
struct vector3 {
	float x, y, z;
};

struct colorVector {
	float r, g, b;
	float a = 1; // Default to have alpha be 1
};

#endif //ERPT_RENDER_ENGINE_TYPES_H
