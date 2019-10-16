

//Using SDL and standard IO
#include <SDL.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <ctime> 
#include <thread>
#include <atomic>
#include <future>
#include <algorithm>



//Screen dimension constants
const int SCREEN_WIDTH = 640;
const int SCREEN_HEIGHT = 380;
//const int SCREEN_WIDTH = 1200;
//const int SCREEN_HEIGHT = 768;
//const int SCREEN_WIDTH = 1920;
//const int SCREEN_HEIGHT = 1080;
const int MAX_DEPTH = 3;

const double PI = 3.14159265358979323846; 



class Vec3 {
public:
	double x, y, z;
	Vec3() {
		x = 0; y = 0; z = 0;
	}
	Vec3(double n) {
		x = n; y = n; z = n;
	}
	Vec3(double xx, double yy, double zz) {
		x = xx; y = yy; z = zz;
	}

	Vec3 operator - (const Vec3& v) const {
		return Vec3(x - v.x, y - v.y, z - v.z);
	}

	Vec3 operator + (const Vec3& v) const {
		return Vec3(x + v.x, y + v.y, z + v.z);
	}

	Vec3 operator * (const double& xn) const {
		return Vec3(xn * x, xn * y, xn * z);
	}

	Vec3 operator * (const Vec3& v) const {
		return Vec3(v.x * x, v.y * y, v.z * z);
	}

	double length() const {
		return sqrt(length2());
	}

	double length2() const {
		return dot(*this);
	}

	double dot(const Vec3& v) const {
		return v.x * x + v.y * y + v.z * z;
	}

	Vec3 cross(const Vec3& v) const {
		double xn = y * v.z - z * v.y;
		double yn = z * v.x - x * v.z;
		double zn = x * v.y - y * v.x;
		return Vec3(xn, yn, zn);
	}
	Vec3& normalize() {
		double l = length();
		x /= l;
		y /= l;
		z /= l;
		return *this;
	}

	Vec3 newnormalize() {
		double l = length();
		return Vec3(x / l, y / l, z / l);
	}

	Vec3& operator = (const Vec3& v) {
		x = v.x;
		y = v.y;
		z = v.z;
		return *this;
	}
	/*Vector.yrot = function(v,a) {
	// y_rotated = ((y_origin - y) * cos(angle)) - ((x - x_origin) * sin(angle)) + y_origin
	cosa = Math.cos(a);
	sina = Math.sin(a);
	return [v[0]*cosa - v[2]*sina, v[1], v[0]*sina + v[2]*cosa];
}*/

	Vec3 rotatey(double a) {
		double cosa = cos(a);
		double sina = sin(a);
		return Vec3((x * cosa) - (z * sina), y, (x * sina) - (z * cosa));
	}

	Vec3 copy() {
		return Vec3(x, y, z);
	}

	Vec3 reflect(const Vec3& I, const Vec3& N)
	{
		//return I - 2 * dotProduct(I, N) * N;
		return (I - (N * (2 * I.dot(N))));
	}

	Vec3 reflect_this(const Vec3& N) {
		double dubdot = 2 * (*this).dot(N);
		return (*this) - (N * (dubdot));
	}

	void print() const {
		printf("X: %f Y: %f Z: %f\n",x,y,z);
	}
};




struct Color {
	double r, g, b;
	Color() : r(0), g(0), b(0) {}
	Color(double c) : r(c), g(c), b(c) {}
	Color(double _r, double _g, double _b) : r(_r), g(_g), b(_b) {}
	Color operator * (double f) const { return Color(r * f, g * f, b * f); }
	Color operator * (Vec3 f) const { return Color(r * f.x, g * f.y, b * f.z); }
	Color operator * (Color c) const { return Color(r * c.r, g * c.g, b * c.b); }
	Color operator + (Color c) const { return Color(r + c.r, g + c.g, b + c.b); }
	Color& operator += (const Color& c) { r += c.r, g += c.g, b += c.b; return *this; }
	Color& operator *= (const Color& c) { r *= c.r, g *= c.g, b *= c.b; return *this; }

	Color& clamp() {
		if (r < 0) r = 0;
		if (g < 0) g = 0;
		if (b < 0) b = 0;
		if (r > 255) r = 255;
		if (g > 255) g = 255;
		if (b > 255) b = 255;
		return *this;
	}


};





class Ray {
public:
	Vec3 o;
	Vec3 d;
	Ray() {
		Vec3 o = Vec3(0, 0, 0);
		Vec3 d = Vec3(0, 0, 1);
	}
	Ray(Vec3& O, Vec3& D) {
		o = O;
		d = D;
	}
	void print() {
		printf("\n\nRay\n===\n\n");
		printf("O: "); o.print();
		printf("D: "); d.print();
	}
};

class Light {
public:
	Vec3 pos;
	Light() {
		pos = Vec3(10, 10, -10);
	}
	Light(Vec3& POS) {
		pos = POS;
	}
};

class Camera {
public:
	Vec3 o;
	Vec3 d;
	double fov;
	Camera() {
		o = Vec3(0, 0, 0);
		d = Vec3(0, 0, 1);
		fov = 90.0;
	}
};


class Sphere; // forward declaration of Sphere to work in Hit class


class Hit {
public:
	Vec3 pt;
	double dist;
	Sphere* object;
	Vec3 normal;
	Hit() {
		pt = Vec3(0, 0, 0);
		dist = INFINITY;
		object = NULL;
		normal = Vec3(0, 0, 1);
	}
	Hit(Vec3& PT, double DIST, Sphere* OBJECT, Vec3& NORMAL) {
		pt = PT;
		dist = DIST;
		object = OBJECT;
		normal = NORMAL;
	}
};

class Mat {
public:
	Color color;
	double reflect;
	double shine;
	Mat() {
		color = Color(255, 0, 0);
		reflect = 0;
		shine = 0;
	}
	Mat(Color& COLOR) {
		color = COLOR;
		reflect = 0;
		shine = 0;
	}
	Mat(Color& COLOR, double REFLECT) {
		color = COLOR;
		reflect = REFLECT;
		shine = REFLECT;
	}
	Mat(Color& COLOR, double REFLECT, double SHINE) {
		color = COLOR;
		reflect = REFLECT;
		shine = SHINE;
	}
	void print() {
		printf("\nMaterial:\n========\nColor: ");
		//color.print();
		printf("\nReflect: %f\n", reflect);
	}
};

class Entity {
public:
	Mat* mat;
	Vec3 pos;
	Entity() {
		mat = new Mat();
		pos = Vec3(0, 0, 20);
	}
	virtual bool shadow_ray_test(Ray& ray, double dist) {
		return false;
	}
	virtual void ray_intersect(Ray& ray, Hit& hit) {
		printf("ERROR ");
	}
	virtual void print() {
	}
};


class Sphere : public Entity {
public:
	Vec3 pos;
	double r;
	double r2;
	Mat *mat;
	Sphere() {
		pos = Vec3(0, 0, 10);
		r = 2.0;
		r2 = 4.0;
		//color = Vec3(255, 0, 0);
		mat = new Mat();
	}
	Sphere(Vec3 &p, double R) {
		pos = p;
		r = R;
		r2 = R * R;
		//color = Vec3(255, 0, 0);
		mat = new Mat();
	}

	bool shadow_ray_test(Ray& ray, double dist) override {
		Vec3 e2c = pos - ray.o;
		double v = e2c.dot(ray.d);
		double eoDot = e2c.dot(e2c);
		double discriminant = (r2)-eoDot + (v * v);
		if (discriminant >= 0) {
			double this_dist = v - sqrt(discriminant);
			if (this_dist>0 && this_dist < dist) {
				// this obj is between source and light @ dist away
				//Vec3 pt = ray.o + (ray.d * dist);
				return true;
			}
			
		}
		return false;
		
	}
	
	Vec3 *get_normal(Vec3& a) {
		return (&(a - pos).normalize());
	}

	void ray_intersect(Ray &ray, Hit &hit) override {
		// DIST is distance to closest entity checked so far - nominally INFINITY
		Vec3 e2c = pos - ray.o;
		double v = e2c.dot(ray.d);
		double eoDot = e2c.dot(e2c);
		double discriminant = (r2) - eoDot + (v*v);
		if (discriminant > 0) {
			double dist = v - sqrt(discriminant);
			if (dist>0 && dist < hit.dist) {
				hit.dist = dist;
				hit.pt = ray.o + (ray.d * dist);
				hit.normal = (hit.pt - pos).normalize();
				hit.object = this;
			}
		}
	}
	void print() override {
		printf("\n\nSphere\n======\n\nPos: ");
		pos.print();
		printf("Radius: %f\n", r);
		mat->print();
	}
};



class Scene {
public:
	Camera camera;
	std::vector<Light*> lights;
	std::vector<Entity*> entities;
	double ambient;
	Scene() {
		ambient = 0.05;
	}
	void print() {
		printf("\n\nSCENE\n=====\n\n");
		std::vector<Entity*>::iterator iter, end;
		for (iter = entities.begin(), end = entities.end(); iter != end; ++iter) {
			(*iter)->print();
		}
		system("PAUSE");
	}
};




Scene scene;



Color trace(Ray &ray, int depth) {

	Color final_color;

	std::vector<Entity*>::iterator iter, end;
	double dist = INFINITY;
	Entity* closest = NULL;
	/*ray.print();
	system("PAUSE");*/
	Hit hit;
	hit.dist = INFINITY;
	/*for (iter = scene.entities.begin(), end = scene.entities.end(); iter != end; ++iter) {
		(*iter)->ray_intersect(ray, hit);
		if (hit.dist < dist) {
			closest = (*iter);
			dist = hit.dist;
		}
	}*/
	for (int l = 0; l < scene.entities.size(); l++) {
		scene.entities[l]->ray_intersect(ray, hit);
	}
	if (hit.dist < INFINITY && dist>0.00000001) {
		//hit.normal = *hit.object->get_normal(hit.pt);
		double light_amount = 0.0;
		double lambertAmount = 0.0;
		double specularAmount = 0.0;
		/*
		for (l=0; l<localscene.lights.length; l++) {
			if (SHADOWS) {
				// vector: Vector.unitVector(Vector.subtract(pt, light)
				var vector_to_light = Vector.subtract( localscene.lights[l].pos, hitpt );
				var shadow_ray_dir = Vector.normalize(vector_to_light);
				//var shadow_ray = new Ray(hitpt, shadow_ray_dir);
				var shadow_ray = {o:hitpt, d:shadow_ray_dir};
				// get dist to light - light may be between
				// hit object and distant object, so not in shadow
				var dist_to_light = Vector.length(vector_to_light);
				//if (shadow_test_ignoreself(shadow_ray, localscene, hitobj)) {
				if (shadow_test(shadow_ray, localscene, dist_to_light)) {
					continue;
				}
			}
			// do light contribution
			var contribution = Vector.dot(Vector.normalize(Vector.subtract(localscene.lights[l].pos, hitpt)), normal);
			if (contribution > 0) {
				lambertAmount += contribution;
			}
		}
		*/
		double contribution;
		bool blocked = false;
		for (int l = 0; l < scene.lights.size(); l++) {
			Vec3 vector_to_light = scene.lights[l]->pos - hit.pt;
			Vec3 shadow_ray_dir = vector_to_light.newnormalize();
			Vec3 shadow_ray_o = hit.pt + (hit.normal * 0.0000001);
			Ray shadow_ray(shadow_ray_o, shadow_ray_dir);
			double dist_to_light = vector_to_light.length();
			// test shadow rays against all scene TODO
			blocked = false;
			for (int o=0; o < scene.entities.size(); o++) {
				blocked = scene.entities[o]->shadow_ray_test(shadow_ray, dist_to_light);
				if (blocked) {
					break;
				}
			}
			if (!blocked) {
				vector_to_light.normalize();
				contribution = vector_to_light.dot(hit.normal);
				lambertAmount += contribution;
				

					/*
					if nearestobject.mat.spec_level > 0:
							fudged_hitpoint = vec_add (nearesthitpoint, vec_div (norm,1000000.0) )
							c = vec_dot (norm, ray.d)
							c = -c
							reflection_dir = vec_add (ray.d, vec_mul (vec_mul(norm,c),2) )
							#reflection_ray = Ray (fudged_hitpoint, reflection_dir)
							V = ray.d
							R = reflection_dir
							L = shadow_ray.d
							vrdot = vec_dot ( V, R )
							dot = vec_dot ( L, R )
							if dot>0:
								dot = dot ** nearestobject.mat.spec_size
								spec_strength = dot * nearestobject.mat.spec_level * light_normal
								spec = colour_scalar_mul ( light.colour, spec_strength )
							else:
								spec = (0,0,0)
							final_colour = colour_add ( final_colour, spec )
					*/
				if (hit.object->mat->reflect > 0) {
					Vec3 c = hit.normal.dot(ray.d);
					c = c * -1.0;
					Vec3 reflection_dir = ray.d.reflect_this(hit.normal);
					/*Vec3 V = ray.d;
					Vec3 R = reflection_dir;
					Vec3 L = shadow_ray.d;*/
					double vrdot = ray.d.dot(reflection_dir);
					double dot = shadow_ray.d.dot(reflection_dir);
					if (dot > 0) {
						dot = pow(dot, hit.object->mat->shine*250);
						//dot = pow(dot, 145);
						specularAmount += dot * hit.object->mat->shine ;
					}
				}
				
			}
		}
		//lambertAmount = (lambertAmount < 1.0) ? lambertAmount : 1.0;
		lambertAmount = (lambertAmount > scene.ambient) ? lambertAmount : scene.ambient;
		
		if (depth < MAX_DEPTH && hit.object->mat->reflect > 0.0) {
			Color lambert_contribution = hit.object->mat->color * lambertAmount * (1 - hit.object->mat->reflect);
			Color specular_contribution = (Color(255, 255, 255) * specularAmount).clamp() * (1 - hit.object->mat->reflect);
	
			/*Vec3 I = ray.d;
			Vec3 N = hit.normal; 
			double cosI = N.dot(I);
			Vec3 dubn = N * (2 * cosI);*/
			//Vec3 R = (I - N * 2 * cosI).normalize();
			Vec3 R = (ray.d - hit.normal * 2 * hit.normal.dot(ray.d));
			Vec3 O = hit.pt + (hit.normal * 0.00000001);
			Ray rRay(O, R );

			/*ray.print();
			printf(" x: %d y: %d\n", x, y);
			system("PAUSE");*/
			/*rRay.print();
			system("PAUSE");*/

			Color perfect_reflection_contribution = trace(rRay, depth + 1) ;
			final_color = lambert_contribution + (perfect_reflection_contribution* hit.object->mat->reflect) + specular_contribution;
			final_color.clamp();
			//final_color = lambert_contribution + (perfect_reflection_contribution * hit.object->mat->reflect);
			//final_color = Color(255, 255, 255);
			//final_color = perfect_reflection_contribution;
			//final_color = hit.object->mat->color * lambertAmount;
		}
		else {
			final_color = hit.object->mat->color * lambertAmount;
		}
		return (final_color);
	}
	else {
		
		
		double ambient = 255.0 * scene.ambient;
		if (ray.d.y < 0) {
			/*final_color.r = 10.0;
			final_color.g = 30.0;
			final_color.b = 10.0;*/
			final_color.r = ambient;
			final_color.g = ambient;
			final_color.b = ambient;
		}
		else {
			double blue = ray.d.y * 255;
			final_color.r = 255.0 * scene.ambient;
			final_color.g = 255.0 * scene.ambient;
			final_color.b = blue;
		}
		return (final_color);
	}
}



int main(int argc, char* args[])
{
	srand((unsigned)time(0));

	Vec3 pos = Vec3(0, 0, 10);
	Sphere *s1 = new Sphere(pos,2.0);
	s1->mat->color = Color(100, 100, 255);
	s1->mat->reflect = 0.3;
	s1->mat->shine = 0.99;
	scene.entities.push_back(s1);

	Vec3 pos2 = Vec3(3, 3, 10);
	Sphere* s2 = new Sphere(pos2,3.0);
	s2->mat->reflect = 0.3;
	s2->mat->shine = 0.3;
	s2->mat->color = Color(255, 100, 100);
	scene.entities.push_back(s2);

	Vec3 pos3 = Vec3(0, -100, 15);
	Sphere* s3 = new Sphere(pos3, 99.0);
	s3->mat->reflect = 0.5;
	s3->mat->shine = 0.5;
	s3->mat->color = Color(22, 55, 22);
	scene.entities.push_back(s3);

	/*Sphere* spheres = new Sphere[10];
	for (int i = 0; i < 10; i++) {
		spheres[i].pos.x = -10+rand()%20;
		spheres[i].pos.y = -10+rand()%20;
		spheres[i].pos.z = -10+rand()%20;
		spheres[i].r = rand() % 10;
		spheres[i].mat->reflect = 0.2;
		scene.entities.push_back(&spheres[i]);
	}
	 */

	Light *l1 = new Light();
	l1->pos = Vec3(-20, 20, 5);
	scene.lights.push_back(l1);

	/*Light* l2 = new Light();
	l2->pos = Vec3(10, 20, 55);
	scene.lights.push_back(l2);*/

	scene.camera.fov = 90.0;

	Vec3 UP = Vec3(0, -1, 0);
	Vec3 vpRight = scene.camera.d.cross(UP).normalize();
	Vec3 vpUp = vpRight.cross(scene.camera.d).normalize();


	double fovRadians = PI * (scene.camera.fov / 2.0) / 180.0;
	double heightWidthRatio = (double)SCREEN_HEIGHT / (double)SCREEN_WIDTH;
	double halfWidth = tan(fovRadians);
	double halfHeight = heightWidthRatio * halfWidth;
	double camerawidth = halfWidth * 2.0;
	double cameraheight = halfHeight * 2.0;
	double pixelWidth = camerawidth / (SCREEN_WIDTH - 1.0);
	double pixelHeight = cameraheight / (SCREEN_HEIGHT - 1.0);

	//The window we'll be rendering to
	SDL_Window* window = NULL;
	SDL_Renderer* renderer = NULL;

	//The surface contained by the window
	SDL_Surface* screenSurface = NULL;
	
	

	//Initialize SDL
	if (SDL_Init(SDL_INIT_VIDEO) < 0)
	{
		printf("SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
	}
	else
	{
		SDL_CreateWindowAndRenderer(SCREEN_WIDTH, SCREEN_HEIGHT, 0, &window, &renderer);
		//SDL_SetWindowFullscreen(window, true);
		SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, SCREEN_WIDTH, SCREEN_HEIGHT);

		// LOOP
		

		SDL_SetRenderDrawColor(renderer, 0, 0, 0, 0);
		SDL_RenderClear(renderer);

		std::vector< unsigned char > pixels(SCREEN_WIDTH * SCREEN_HEIGHT * 4, 0);

		//scene.print();

		

		std::size_t max = SCREEN_WIDTH * SCREEN_HEIGHT;
		//std::size_t cores = std::thread::hardware_concurrency();
		
		std::vector<std::future<void>> future_vector;

		int SCREEN_WIDTH_TIMES_FOUR = SCREEN_WIDTH * 4;

		volatile std::atomic<std::size_t> count(0);

		while (true) {
			//memset(pixels, 255, 640 * 480 * sizeof(Uint32));
			//SDL_UpdateTexture(texture, NULL, pixels, 640 * sizeof(Uint32));



			//printf(".");
			int start = SDL_GetTicks();
			
			//l1->pos = l1->pos.rotatey(l1_rotate);
			//l1_rotate += 0.001;
			//l2->pos.x -= l1_rotate*2;
			scene.entities[1]->pos.x -= 0.01;
			printf("X: %f", scene.entities[1]->pos.x);
			

			//while (cores--)
				//future_vector.emplace_back(
				//	std::async([=, &count, &pixels]()
				//		{
				//			while (true)
				//			{
				//				std::size_t index = count++;
				//				if (index >= max) {}
				//					break;
				//				std::size_t x = index % SCREEN_WIDTH;
				//				std::size_t y = index / SCREEN_WIDTH;
				//				// make ray etc
				//				Ray ray;
				//				Vec3 xcomp = vpRight * ((x * pixelWidth) - halfWidth);
				//				Vec3 ycomp = vpUp * ((y * pixelHeight) - halfHeight);
				//				ray.d = ((scene.camera.d + xcomp) + ycomp).normalize();
				//				Color color = trace(ray, 1);
				//				const unsigned int offset = (SCREEN_WIDTH_TIMES_FOUR * y) + x * 4;
				//				pixels[offset + 0] = color.b;        // b
				//				pixels[offset + 1] = color.g;        // g
				//				pixels[offset + 2] = color.r;        // r
				//				pixels[offset + 3] = SDL_ALPHA_OPAQUE;    // a
				//			}
				//		}));
			count = 0;
			while (count < max) {
				#pragma omp parallel for 
				for (int x = 0; x < SCREEN_WIDTH; x++) {
					for (int y = 0; y < SCREEN_HEIGHT; y++) {
						Color color;
						Ray ray;
						ray.o = scene.camera.o;
						Vec3 xcomp = vpRight * ((x * pixelWidth) - halfWidth);
						Vec3 ycomp = vpUp * ((y * pixelHeight) - halfHeight);
						ray.d = ((scene.camera.d + xcomp) + ycomp).normalize();
						color = trace(ray, 1);
						const unsigned int offset = (SCREEN_WIDTH_TIMES_FOUR * y) + x * 4;
						pixels[offset + 0] = color.b;        // b
						pixels[offset + 1] = color.g;        // g
						pixels[offset + 2] = color.r;        // r
						pixels[offset + 3] = SDL_ALPHA_OPAQUE;    // a
						count++;
					}
				};
			}

			

			
			
				SDL_UpdateTexture
				(
					texture,
					NULL,
					&pixels[0],
					SCREEN_WIDTH * 4
				);
				SDL_RenderCopy(renderer, texture, NULL, NULL);
				SDL_RenderPresent(renderer);

				int ticks = SDL_GetTicks() - start;
				float fps = 1000.0f / (float)ticks;
				printf("\n%f", fps);
				SDL_PumpEvents();
			
			
		}

		//SDL_DestroyTexture(texture);
	}

	

	//Destroy window
	SDL_DestroyWindow(window);

	//Quit SDL subsystems
	SDL_Quit();

	return 0;
}
