#ifndef RT_ENGINE_OBJECT_CLASS_H
#define RT_ENGINE_OBJECT_CLASS_H

namespace RT_ENGINE {

class Object {
public:
	virtual void Start(float time) = 0;
	virtual void Update(float time, float deltaTime) = 0;
};

} // namespace RT_ENGINE //

#endif // ifndef RT_ENGINE_OBJECT_CLASS_H //