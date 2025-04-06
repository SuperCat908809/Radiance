#include "world.h"

using namespace RT_ENGINE;

float World::getTime() const { return _time; }
float World::getDeltaTime() const { return _deltaTime; }

void World::Start() {
	for (Object* obj : objects) {
		obj->Start(_time);
	}
}
void World::Update() {
	for (Object* obj : objects) {
		obj->Update(_time, _deltaTime);
	}
}

void World::AddObject(Object* obj) {
	if (std::find(objects.begin(), objects.end(), obj) == objects.end())
		return;
	objects.push_back(obj);
}
void World::RemoveObject(Object* obj) {
	objects.erase(std::find(objects.begin(), objects.end(), obj));
}
const std::vector<Object*>& World::getObjects() const { 
	return objects;
}