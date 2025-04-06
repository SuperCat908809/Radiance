#ifndef RT_ENGINE_WORLD_CLASS_H
#define RT_ENGINE_WORLD_CLASS_H

#include <vector>
#include "object.h"


namespace RT_ENGINE {

class World {
	float _time{ 0.0f }, _deltaTime{ 1.0f };

	std::vector<Object*> objects{};
	// essentially an event manager so all objects can be updated from one place.

	// world material
	// scene object? abstract class for scene?

public:
	float getTime() const;
	float getDeltaTime() const;

	void Start();
	void Update();

	void AddObject(Object* obj);
	void RemoveObject(Object* obj);
	const std::vector<Object*>& getObjects() const;
};
}

#endif // ifndef RT_ENGINE_WORLD_CLASS_H //