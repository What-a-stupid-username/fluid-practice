//88-Line 2D Moving Least Squares Material Point Method (MLS-MPM)[with comments]
//#define TC_IMAGE_IO   // Uncomment this line for image exporting functionality
#include "taichi.h"    // Note: You DO NOT have to install taichi or taichi_mpm.
#include "kernel.cuh"

using namespace taichi;// You only need [taichi.h] - see below for instructions.

namespace original
{

	const int n = 160 /*grid resolution (cells)*/, window_size = 800;
	const real dt = 1e-5_f, frame_dt = 1e-3_f, dx = 1.0_f / n, inv_dx = 1.0_f / dx;
	auto particle_mass = 1.0_f, vol = 1.0_f;
	auto hardening = 10.0_f, E = 1e4_f, nu = 0.2_f;
	real mu_0 = E / (2 * (1 + nu)), lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu));
	using Vec = Vector2; using Mat = Matrix2; bool plastic = true;
	struct Particle {
		Vec x, v; Mat F, C; real Jp; int c/*color*/;
		Particle(Vec x, int c, Vec v = Vec(0)) : x(x), v(v), F(1), C(0), Jp(1), c(c) {}
	};
	std::vector<Particle> particles;
	Vector3 grid[n + 1][n + 1];          // velocity + mass, node_res = cell_res + 1

	void advance(real dt) {
		std::memset(grid, 0, sizeof(grid));                              // Reset grid
		for (auto& p : particles) {                                             // P2G
			Vector2i base_coord = (p.x * inv_dx - Vec(0.5_f)).cast<int>();//element-wise floor
			Vec fx = p.x * inv_dx - base_coord.cast<real>();
			// Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
			Vec w[3]{ Vec(0.5) * sqr(Vec(1.5) - fx), Vec(0.75) - sqr(fx - Vec(1.0)),
					 Vec(0.5) * sqr(fx - Vec(0.5)) };
			auto e = std::exp(hardening * (1.0_f - p.Jp)), mu = mu_0 * e, lambda = lambda_0 * e;
			real J = determinant(p.F);         //                         Current volume
			Mat r, s; polar_decomp(p.F, r, s); //Polar decomp. for fixed corotated model
			auto stress =                           // Cauchy stress times dt and inv_dx
				-4 * inv_dx * inv_dx * dt * vol * (2 * mu * (p.F - r) * transposed(p.F) + lambda * (J - 1) * J);
			auto affine = stress + particle_mass * p.C;
			for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) { // Scatter to grid
				auto dpos = (Vec(i, j) - fx) * dx;
				Vector3 mv(p.v * particle_mass, particle_mass); //translational momentum
				grid[base_coord.x + i][base_coord.y + j] +=
					w[i].x * w[j].y * (mv + Vector3(affine * dpos, 0));
			}
		}
		#pragma omp parallel for
		for (int i = 0; i <= n; i++) for (int j = 0; j <= n; j++) { //For all grid nodes
			auto& g = grid[i][j];
			if (g[2] > 0) {                                // No need for epsilon here
				g /= g[2];                                   //        Normalize by mass
				g += dt * Vector3(0, -200, 0);               //                  Gravity
				real boundary = 0.05, x = (real)i / n, y = real(j) / n; //boundary thick.,node coord
				if (x < boundary || x > 1 - boundary || y > 1 - boundary) g = Vector3(0); //Sticky
				if (y < boundary) g[1] = std::max(0.0_f, g[1]);             //"Separate"
			}
		}
		#pragma omp parallel for
		for (int pidx = 0; pidx < particles.size(); pidx++) {                                // Grid to particle
			auto& p = particles[pidx];
			Vector2i base_coord = (p.x * inv_dx - Vec(0.5_f)).cast<int>();//element-wise floor
			Vec fx = p.x * inv_dx - base_coord.cast<real>();
			Vec w[3]{ Vec(0.5) * sqr(Vec(1.5) - fx), Vec(0.75) - sqr(fx - Vec(1.0)),
					 Vec(0.5) * sqr(fx - Vec(0.5)) };
			p.C = Mat(0); p.v = Vec(0);
			for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) {
				auto dpos = (Vec(i, j) - fx),
					grid_v = Vec(grid[base_coord.x + i][base_coord.y + j]);
				auto weight = w[i].x * w[j].y;
				p.v += weight * grid_v;                                      // Velocity
				p.C += 4 * inv_dx * Mat::outer_product(weight * grid_v, dpos); // APIC C
			}
			p.x += dt * p.v;                                                // Advection
			auto F = (Mat(1) + dt * p.C) * p.F;                      // MLS-MPM F-update
			Mat svd_u, sig, svd_v; svd(F, svd_u, sig, svd_v);
			for (int i = 0; i < 2 * int(plastic); i++)                // Snow Plasticity
				sig[i][i] = clamp(sig[i][i], 1.0_f - 2.5e-2_f, 1.0_f + 7.5e-3_f);
			real oldJ = determinant(F); F = svd_u * sig * transposed(svd_v);
			real Jp_new = clamp(p.Jp * oldJ / determinant(F), 0.6_f, 20.0_f);
			p.Jp = Jp_new; p.F = F;
		}
	}
	void add_object(Vec center, int c) {   // Seed particles with position and color
		for (int i = 0; i < 500; i++)  // Randomly sample 1000 particles in the square
			particles.push_back(Particle((Vec::rand() * 2.0_f - Vec(1)) * 0.08_f + center, c));
	}
	int main() {
		GUI gui("Real-time 2D MLS-MPM", window_size, window_size);
		add_object(Vec(0.55, 0.45), 0xED553B); add_object(Vec(0.45, 0.65), 0xF2B134);
		add_object(Vec(0.55, 0.85), 0x068587); auto& canvas = gui.get_canvas(); int f = 0;
		for (int i = 0;; i++) {                              //              Main Loop
			advance(dt);                                       //     Advance simulation
			if (i % int(frame_dt / dt) == 0) {                 //        Visualize frame
				canvas.clear(0x112F41);                          //       Clear background
				canvas.rect(Vec(0.04), Vec(0.96)).radius(2).color(0x4FB99F).close();// Box
				for (auto p : particles)canvas.circle(p.x).radius(2).color(p.c);//Particles
				gui.update();                                              // Update image
				// canvas.img.write_as_image(fmt::format("tmp/{:05d}.png", f++));
			}
		}
	}
}

namespace MAC {

	const int GridCount = 256;
	const real dt = 1e-4_f, frame_dt = 1e-3_f, dx = 1.0_f / GridCount, inv_dx = 1.0_f / dx;

	Vector2 Velocity[GridCount + 1][GridCount + 1];
	Vector2 Velocity_[GridCount + 1][GridCount + 1];

	float Color[GridCount][GridCount];
	float Color_[GridCount][GridCount];

	float Pressure[GridCount][GridCount];

	float lerp(float a, float b, float c) {
		return a * (1 - c) + b * c;
	}

	Vector2& SRV(Vector2 q[GridCount + 1][GridCount + 1], int x, int y) {
		//return Velocity[x][y];
		return q[min(GridCount, max(0, x))][min(GridCount - 1, max(0, y))];
	}

	Vector2 CenterSpeed(Vector2 q[GridCount + 1][GridCount + 1], Vector2i idx) {
		int i = idx.x;
		int j = idx.y;
		float vx = SRV(q, i, j).x + SRV(q, i + 1, j).x; vx /= 2;
		float vy = SRV(q, i, j).y + SRV(q, i, j + 1).y; vy /= 2;
		return Vector2(vx, vy);
	}

	Vector2 EdgeSpeedX(Vector2 q[GridCount + 1][GridCount + 1], Vector2i idx) {
		int i = idx.x;
		int j = idx.y;
		return Vector2(SRV(q, i, j).x, (SRV(q, i - 1, j).y + SRV(q, i - 1, j + 1).y + SRV(q, i, j).y + SRV(q, i, j + 1).y) / 4);
	}

	Vector2 EdgeSpeedY(Vector2 q[GridCount + 1][GridCount + 1], Vector2i idx) {
		int i = idx.x;
		int j = idx.y;
		return Vector2((SRV(q, i, j).x + SRV(q, i + 1, j).x + SRV(q, i, j - 1).x + SRV(q, i + 1, j - 1).x) / 4, SRV(q, i, j).y);
	}

	float EdgeSpeedXX(Vector2 q[GridCount + 1][GridCount + 1], Vector2 idx) {
		float i_ = idx.x;
		float j_ = idx.y;

		int i = i_;
		int j = j_;
		float fi = i_ - i;
		float fj = j_ - j;

		return lerp(
			lerp(SRV(q, i, j).x, SRV(q, i + 1, j).x, fi),
			lerp(SRV(q, i, j + 1).x, SRV(q, i + 1, j + 1).x, fi),
			fj);
	}

	float EdgeSpeedYY(Vector2 q[GridCount + 1][GridCount + 1], Vector2 idx) {
		float i_ = idx.x;
		float j_ = idx.y;

		int i = i_;
		int j = j_;
		float fi = i_ - i;
		float fj = j_ - j;

		return lerp(
			lerp(SRV(q, i, j).y, SRV(q, i + 1, j).y, fi),
			lerp(SRV(q, i, j + 1).y, SRV(q, i + 1, j + 1).y, fi),
			fj);
	}

	float& SR(float q[GridCount][GridCount], int x, int y) {
		return q[min(GridCount - 1, max(0, x))][min(GridCount - 1, max(0, y))];
	}

	Vector4 Cubic(float s) {
		return Vector4(
			-1.0 / 3 * s + 1.0 / 2 * s * s - 1.0 / 6 * s * s * s,
			1 - s * s + 1.0 / 2 * (s * s * s - s),
			s + 1.0 / 2 * (s * s - s * s * s),
			1.0 / 6 * (s * s * s - s)
		);
	}

	float Q(float q[GridCount][GridCount], Vector2 idx) {

		float i_ = idx.x;
		float j_ = idx.y;

		int i = i_;
		int j = j_;
		float fi = i_ - i;
		float fj = j_ - j;

		Vector4 cubicx = Cubic(fi);
		Vector4 cubicy = Cubic(fj);
		//float cubicx[4] = { 0, 1 - fi, fi, 0 };
		//float cubicy[4] = { 0, 1 - fj, fj, 0 };

		float res = 0;
		for (int ox = -1; ox <= 2; ox++) for (int oy = -1; oy <= 2; oy++)
		{
			res += SR(q, i + ox, j + oy) * cubicx[ox + 1] * cubicy[oy + 1];
		}
		return max(0.0f, res);
	}

	void Init() {
		for (int i = 0; i <= GridCount; i++) for (int j = 0; j <= GridCount; j++)
		{
			Velocity[i][j] = Vector2(0);
		}
		for (int i = 0; i < GridCount; i++) for (int j = 0; j < GridCount; j++)
		{
			Pressure[i][j] = 0;
			//float u = (float)i / GridCount - 0.5;
			//float v = (float)j / GridCount - 0.5;
			//float val = max(0.0f, sqrt(u * u + v * v) * 2) < 0.8 ? 1.0f : 0.0f;
			float u = (float)i / GridCount;
			float v = (float)j / GridCount;
			float val = u > 0.1 && u < 0.25 && v > 0.5 && v < 0.65;
			Color[i][j] = val;
		}
	}

	bool Solid(int i, int j) {
		real boundary = 0.04, x = (real)i / GridCount, y = real(j) / GridCount;

		if (x < boundary || y < boundary || x > 1 -boundary) return true;

		//if ((x - 0.5) * (x - 0.5) + (y - 0.85) * (y - 0.85) < 0.001) return true;

		return false;
	}

	bool Air(int i, int j) {
		real boundary = 0.04, x = (real)i / GridCount, y = real(j) / GridCount;

		//if (y < boundary) return true;
		////if (x > 1 - boundary) return true;
		//return false;

		//if (i < 0 || j < 0 || i >= GridCount || j >= GridCount) return true;
		if (Color[i][j] == 0) return true;
		return false;
	}

	void Simulate(float deltaT) {

		// Advect
		{
			#pragma omp parallel for
			for (int i = 0; i < GridCount; i++) for (int j = 0; j < GridCount; j++)
			{
				Vector2 velocity = CenterSpeed(Velocity, Vector2i(i, j));

				Vector2 ij = Vector2(i, j) - velocity * deltaT;

				Color_[i][j] = Q(Color, ij);
				//if (j >= GridCount - 10)
					//Color_[i][j] = int(((float)i / GridCount) * 43) % 2;
			}
			memcpy(Color, Color_, sizeof(Color));

			#pragma omp parallel for
			for (int i = 0; i <= GridCount; i++) for (int j = 0; j <= GridCount; j++)
			{
				Vector2 vx = EdgeSpeedX(Velocity, Vector2i(i, j));
				Vector2 vy = EdgeSpeedY(Velocity, Vector2i(i, j));
				Vector2 ij = Vector2(i, j) - vx * deltaT;
				Velocity_[i][j].x = EdgeSpeedXX(Velocity, ij);
				ij = Vector2(i, j) - vy * deltaT;
				Velocity_[i][j].y = EdgeSpeedYY(Velocity, ij) - 200 * GridCount * deltaT;

				//if (j >= GridCount - 10)
					//Velocity_[i][j].y = -20 * GridCount;

				if (Solid(i, j)) {
					Velocity_[i][j] = Vector2(0);
				}
				else {
					if (Solid(i - 1, j)) {
						Velocity_[i][j].x = 0;
					}
					if (Solid(i, j - 1)) {
						Velocity_[i][j].y = 0;
					}
				}
			}
		}

		// project
		int total_iter = 40;
		for (int loop = 0; loop < total_iter; loop++)
		{
			#pragma omp parallel for
			for (int i = 0; i <= GridCount; i++) for (int j = 0; j <= GridCount; j++)
			{
				Vector2 xy = Velocity_[i][j];
				Vector2 deltaV = Vector2(SR(Pressure, i - 1, j) - SR(Pressure, i, j), SR(Pressure, i, j - 1) - SR(Pressure, i, j));
				real boundary = 0.04, x = (real)i / GridCount, y = real(j) / GridCount;

				if (Solid(i, j)) {
					deltaV = Vector2(0);
				}
				else {
					if (Solid(i - 1, j)) {
						deltaV.x = 0;
					}
					if (Solid(i, j - 1)) {
						deltaV.y = 0;
					}
				}
				Velocity[i][j] = xy + deltaV;

				//if (Air(i, j)) {
				//	Velocity[i][j] = max(Vector2(0), Velocity[i][j]);
				//}
				//if (Air(i - 1, j)) {
				//	Velocity[i][j].x = min(0.0f, Velocity[i][j].x);
				//}
				//if (Air(i, j - 1)) {
				//	Velocity[i][j].y = min(0.0f, Velocity[i][j].y);
				//}
			}
			#pragma omp parallel for
			for (int i = 0; i < GridCount; i++) for (int j = 0; j < GridCount; j++)
			{
				if (Air(i, j)) {
					Pressure[i][j] = 0;
					continue;
				}

				Vector2 xy = Velocity[i][j];
				float lx = xy.x;
				float dy = xy.y;
				float rx = SRV(Velocity, i + 1, j).x;
				float uy = SRV(Velocity, i, j + 1).y;

				float noSolid = 4;

				noSolid -= Solid(i - 1, j) + Solid(i, j - 1) + Solid(i + 1, j) + Solid(i, j + 1);


				Pressure[i][j] += (lx - rx + dy - uy) / noSolid;
			}
		}

		//memcpy(Velocity, Velocity_, sizeof(Velocity));
	}

	int main() {

		//original::main();

		GUI gui("Real-time 2D MLS-MPM", GridCount * 4, GridCount * 4);

		Init();

		auto& canvas = gui.get_canvas(); int f = 0;
		for (int i = 0;; i++) {                              //              Main Loop
			Simulate(dt);
			if (i % int(frame_dt / dt) == 0) {                 //        Visualize frame
				canvas.clear(0x112F41);                          //       Clear background

				#pragma omp parallel for
				for (int i = 0; i < GridCount; i++) for (int j = 0; j < GridCount; j++)
				{
					Vector2 xy = Velocity[i][j];
					float lx = xy.x;
					float dy = xy.y;
					float rx = SRV(Velocity, i + 1, j).x;
					float uy = SRV(Velocity, i, j + 1).y;

					float value;// = length(grid[i][j].v) * inv_dx / dt / 100;
					value = Color[i][j]; (lx - rx + dy - uy);//  Velocity[i][j].x;// Pressure[i][j];// ; ;// -3.5; //
					int c = max(0.0f, min(1.0f, value)) * 255;
					int c2 = max(0.0f, min(1.0f, value)) * 255;// c;
					canvas.circle(Vector2((float)i / GridCount, (float)j / GridCount)).radius(2).color((c << 16) + (c2 << 8) + c);//Particles
				}
				gui.update();                                              // Update image
				// canvas.img.write_as_image(fmt::format("tmp/{:05d}.png", f++));
			}
		}
	}
}

namespace FLIP {

	const int GridCount = GRID_COUNT;
	const real dt = 1e-4_f, frame_dt = 1e-3_f, dx = 1.0_f / GridCount, inv_dx = 1.0_f / dx;

	struct Particle {
		Vector2 x;
		Vector2 v;
		float mass;
	};

	std::vector<Particle> Particles;

	Vector2 Velocity[GridCount + 1][GridCount + 1];
	Vector2 Velocity_[GridCount + 1][GridCount + 1];
	Vector2 Mass[GridCount + 1][GridCount + 1];

	float Density[GridCount][GridCount];
	//float Pressure[GridCount][GridCount];

	float A[GridCount*GridCount][4];
	float B[GridCount*GridCount];
	float padding_A;
	double X[GridCount][GridCount];
	double temp[GridCount][GridCount];
	int index[GridCount * GridCount];
	atomic_int index_count;

	mutex buffer_mutex;

	float& RA(int i, int j) {
		if (j < 0) return padding_A;
		int row, col;
		if (j >= i) {
			col = i;
			row = (j - i) / GridCount + 2 * (j - i) % GridCount;
		}
		else {
			col= j;
			row = (i - j) / GridCount + 2 * (i - j) % GridCount;
		}
		return A[col][row];
	}
	double& RX(int i, int j) {
		return X[min(GridCount - 1, max(0, i))][min(GridCount - 1, max(0, j))];
	}
	double& SRD(double q[GridCount][GridCount], int i, int j) {
		return q[min(GridCount - 1, max(0, i))][min(GridCount - 1, max(0, j))];
	}

	float lerp(float a, float b, float c) {
		return a * (1 - c) + b * c;
	}

	Vector2& SRV(Vector2 q[GridCount + 1][GridCount + 1], int x, int y) {
		//return Velocity[x][y];
		return q[min(GridCount, max(0, x))][min(GridCount - 1, max(0, y))];
	}

	Vector2 CenterSpeed(Vector2 q[GridCount + 1][GridCount + 1], Vector2i idx) {
		int i = idx.x;
		int j = idx.y;
		float a = SRV(Mass, i, j).x, b = SRV(Mass, i + 1, j).x;
		float vx = SRV(q, i, j).x * a + SRV(q, i + 1, j).x * b; vx /= a + b;
		a = SRV(Mass, i, j).y, b = SRV(Mass, i, j + 1).y;
		float vy = SRV(q, i, j).y * a + SRV(q, i, j + 1).y * b; vy /= a + b;
		return Vector2(vx, vy);
	}

	Vector2 EdgeSpeedX(Vector2 q[GridCount + 1][GridCount + 1], Vector2i idx) {
		int i = idx.x;
		int j = idx.y;
		return Vector2(SRV(q, i, j).x, (SRV(q, i - 1, j).y + SRV(q, i - 1, j + 1).y + SRV(q, i, j).y + SRV(q, i, j + 1).y) / 4);
	}

	Vector2 EdgeSpeedY(Vector2 q[GridCount + 1][GridCount + 1], Vector2i idx) {
		int i = idx.x;
		int j = idx.y;
		return Vector2((SRV(q, i, j).x + SRV(q, i + 1, j).x + SRV(q, i, j - 1).x + SRV(q, i + 1, j - 1).x) / 4, SRV(q, i, j).y);
	}

	float EdgeSpeedXX(Vector2 q[GridCount + 1][GridCount + 1], Vector2 idx) {
		float i_ = idx.x;
		float j_ = idx.y;

		int i = i_;
		int j = j_;
		float fi = i_ - i;
		float fj = j_ - j;

		return lerp(
			lerp(SRV(q, i, j).x, SRV(q, i + 1, j).x, fi),
			lerp(SRV(q, i, j + 1).x, SRV(q, i + 1, j + 1).x, fi),
			fj);
	}

	float EdgeSpeedYY(Vector2 q[GridCount + 1][GridCount + 1], Vector2 idx) {
		float i_ = idx.x;
		float j_ = idx.y;

		int i = i_;
		int j = j_;
		float fi = i_ - i;
		float fj = j_ - j;

		return lerp(
			lerp(SRV(q, i, j).y, SRV(q, i + 1, j).y, fi),
			lerp(SRV(q, i, j + 1).y, SRV(q, i + 1, j + 1).y, fi),
			fj);
	}

	float& SR(float q[GridCount][GridCount], int x, int y) {
		return q[min(GridCount - 1, max(0, x))][min(GridCount - 1, max(0, y))];
	}

	Vector4 Cubic(float s) {
		return Vector4(
			-1.0 / 3 * s + 1.0 / 2 * s * s - 1.0 / 6 * s * s * s,
			1 - s * s + 1.0 / 2 * (s * s * s - s),
			s + 1.0 / 2 * (s * s - s * s * s),
			1.0 / 6 * (s * s * s - s)
		);
	}

	Vector3 BSpline(float s) {
		return Vector3(
			(s * s + s + 0.25) / 2,
			0.75 - s * s,
			(s * s - s + 0.25) / 2
		);
	}


	float Q(float q[GridCount][GridCount], Vector2 idx) {

		float i_ = idx.x;
		float j_ = idx.y;

		int i = i_;
		int j = j_;
		float fi = i_ - i;
		float fj = j_ - j;

		Vector4 cubicx = Cubic(fi);
		Vector4 cubicy = Cubic(fj);
		//float cubicx[4] = { 0, 1 - fi, fi, 0 };
		//float cubicy[4] = { 0, 1 - fj, fj, 0 };

		float res = 0;
		for (int ox = -1; ox <= 2; ox++) for (int oy = -1; oy <= 2; oy++)
		{
			res += SR(q, i + ox, j + oy) * cubicx[ox + 1] * cubicy[oy + 1];
		}
		return max(0.0f, res);
	}

	bool Solid(int i, int j) {
		real boundary = 0.04, x = (i + 0.5) / GridCount, y = (j + 0.5) / GridCount;

		if (x < boundary || y < boundary || x > 1 - boundary || y > 1 - boundary) return true;

		//if ((x - 0.5) * (x - 0.5) + (y - 0.85) * (y - 0.85) < 0.001) return true;

		return false;
	}

	bool Air(int i, int j) {
		bool lx = Density[i - 1][j] || Solid(i - 1, j);
		bool rx = Density[i + 1][j] || Solid(i + 1, j);
		bool dy = Density[i][j - 1] || Solid(i, j - 1);
		bool uy = Density[i][j + 1] || Solid(i, j + 1);
		Vector2 m = Mass[i][j];
		float xm = Mass[i + 1][j].x;
		float ym = Mass[i][j + 1].y;
		//&& ( || !Density[i - 1][j] || !Density[i][j + 1] || !Density[i][j - 1])
		//return !Solid(i, j) && (!Density[i][j]);
		//return !Solid(i, j) && (!Density[i][j]) && !(lx || rx || dy || uy);
		return !Solid(i, j) && (m.x == 0 || m.y == 0 || xm == 0 || ym == 0);// && (!Density[i][j])&&;
	}

	void Init() {
		int w = GridCount;
		for (int i = 0; i < w; i++) for (int j = 0; j < 2 * w; j++)
		{
			Vector2 uv = (Vector2(i, j) + Vector2(0.5)) / float(GridCount);

			if (uv.x < 0.35 && uv.y < 0.25/*uv.x < 0.9*/ /*&& uv.y > 0.1 &&*/&& !Solid(i,j) /*&& uv.y>0.5 && uv.x > 0.1*/) {
				for (int p = 0; p < 4; p++)
				{
					Particles.push_back(Particle{ Vector2::rand() - Vector2(0.5) + uv * Vector2(GridCount), Vector2(0), 1 });
				}
			}
		}
		for (int i = 0; i < GridCount; i++) for (int j = 0; j < GridCount; j++)
		{
			X[i][j] = 0;
		}
	}


	void Simulate(float deltaT) {

		// P2G
		{
			memset(Velocity_, 0, sizeof(Velocity_));
			memset(Mass, 0, sizeof(Mass));
			memset(Density, 0, sizeof(Density));
			#pragma omp parallel for
			for (int i = 0; i < Particles.size(); i++)
			{
				Particle& p = Particles[i];

#if 1
				Vector2 posx = p.x - Vector2(0, 0.5);
				Vector2 posy = p.x - Vector2(0.5, 0);
				Vector2i lbx = posx.cast<int>();
				Vector2i lby = posy.cast<int>();
				Vector2 offsetx = posx - lbx.cast<float>();
				Vector2 offsety = posy - lby.cast<float>();

				//buffer_mutex.lock();
				Density[(int)p.x.x][(int)p.x.y] = p.mass;
				//buffer_mutex.unlock();

				for (int oi = 0; oi <= 1; oi++) for (int oj = 0; oj <= 1; oj++)
				{
					Vector2i ix = lbx + Vector2i(oi, oj);
					Vector2i iy = lby + Vector2i(oi, oj);
					Vector2& vx = SRV(Velocity_, ix.x, ix.y);
					Vector2& vy = SRV(Velocity_, iy.x, iy.y);
					Vector2& mx = SRV(Mass, ix.x, ix.y);
					Vector2& my = SRV(Mass, iy.x, iy.y);
					Vector2 weightx = p.mass * Vector2(abs(1 - offsetx.x - oi), abs(1 - offsetx.y - oj));
					Vector2 weighty = p.mass * Vector2(abs(1 - offsety.x - oi), abs(1 - offsety.y - oj));
					buffer_mutex.lock();
					vx.x += weightx.x * weightx.y * p.v.x;
					mx.x += weightx.x * weightx.y;
					vy.y += weighty.x * weighty.y * p.v.y;
					my.y += weighty.x * weighty.y;
					buffer_mutex.unlock();
				}
#else
				Vector2 posx = p.x - Vector2(0, 0.5);
				Vector2 posy = p.x - Vector2(0.5, 0);
				Vector2i lbx = Vector2i(round(posx.x), round(posx.y));
				Vector2i lby = Vector2i(round(posy.x), round(posy.y));
				Vector2 offsetx = posx - lbx.cast<float>();
				Vector2 offsety = posy - lby.cast<float>();

				Vector3 wxx = BSpline(offsetx.x), wxy = BSpline(offsetx.y);
				Vector3 wyx = BSpline(offsety.x), wyy = BSpline(offsety.y);
				lbx -= Vector2i(1);
				lby -= Vector2i(1);
				//buffer_mutex.lock();
				Density[(int)p.x.x][(int)p.x.y] = p.mass;
				//buffer_mutex.unlock();

				for (int oi = 0; oi <= 2; oi++) for (int oj = 0; oj <= 2; oj++)
				{
					Vector2i ix = lbx + Vector2i(oi, oj);
					Vector2i iy = lby + Vector2i(oi, oj);
					Vector2& vx = SRV(Velocity_, ix.x, ix.y);
					Vector2& vy = SRV(Velocity_, iy.x, iy.y);
					Vector2& mx = SRV(Mass, ix.x, ix.y);
					Vector2& my = SRV(Mass, iy.x, iy.y);
					float weightx = wxx[oi] * wxy[oj];
					float weighty = wyx[oi] * wyy[oj];
					buffer_mutex.lock();
					vx.x += weightx * p.v.x;
					mx.x += weightx;
					vy.y += weighty * p.v.y;
					my.y += weighty;
					buffer_mutex.unlock();
				}
#endif
			}
		}

		// change mv 2 v
		#pragma omp parallel for
		for (int i = 0; i <= GridCount; i++) for (int j = 0; j <= GridCount; j++)
		{
			Vector2& v = Velocity_[i][j];
			Vector2& mass = Mass[i][j];
			v.x = mass.x == 0 ? 0 : v.x / mass.x;
			v.y = mass.y == 0 ? 0 : v.y / mass.y;
			//mass = max(Vector2(0.00001), mass);
		}

		#pragma omp parallel for
		for (int i = 0; i <= GridCount; i++) for (int j = 0; j <= GridCount; j++)
		{
			Vector2 v = Velocity_[i][j];
			Vector2 m = Mass[i][j];
			if (m.x == 0 || m.y == 0) {
				Vector2 vx = Vector2(0);
				Vector2 vy = Vector2(0);
				for (int oi = -1; oi <= 1; oi++) for (int oj = -1; oj <= 1; oj++)
				{
					Vector2 mass = SRV(Mass, i + oi, j + oj);
					Vector2 vel = SRV(Velocity_, i + oi, j + oj);
					vx += mass.x != 0 ? Vector2(vel.x, 1) : Vector2(0);
					vy += mass.y != 0 ? Vector2(vel.y, 1) : Vector2(0);
				}
				v.x = m.x == 0 ? (vx.y != 0 ? vx.x / vx.y : 0) : v.x;
				v.y = m.y == 0 ? (vy.y != 0 ? vy.x / vy.y : 0) : v.y;
			}

			v.y -= 9500 * GridCount * deltaT;

			if (Solid(i, j)) v = min(v, Vector2(0));
			else {
				if (Solid(i - 1, j)) v.x = max(0.0f, v.x);
				if (Solid(i, j - 1)) v.y = max(0.0f, v.y);
			}

			Velocity[i][j] = v;
		}

		// project
		{
			memset(A, 0, sizeof(A));
			memset(B, 0, sizeof(B));
			index_count = 0;
			// Generate Matrix
			#pragma omp parallel for
			for (int i = 0; i < GridCount; i++) for (int j = 0; j < GridCount; j++)
			{
				if (Solid(i, j) || Air(i, j)) {
					continue;
				}
				
				int id = index_count.fetch_add(1);
				int col = i + GridCount * j;
				index[id] = col;

				Vector2 xy = Velocity[i][j];
				float lx = xy.x;
				float dy = xy.y;
				float rx = SRV(Velocity, i + 1, j).x;
				float uy = SRV(Velocity, i, j + 1).y;
				B[col] = (lx - rx + dy - uy);

				if (!Solid(i - 1, j)) {
					float scale = 1;// / SRV(Mass, i, j).x;
					if (!Air(i - 1, j)) {
						RA(col, (i - 1) + GridCount * j) = -scale;
					}
					RA(col, col) += scale;
				}
				if (!Solid(i + 1, j)) {
					float scale = 1;// / SRV(Mass, i + 1, j).x;
					if (!Air(i + 1, j)) {
						RA(col, (i + 1) + GridCount * j) = -scale;
					}
					RA(col, col) += scale;
				}
				if (!Solid(i, j - 1)) {
					float scale = 1;// / SRV(Mass, i, j).y;
					if (!Air(i, j - 1)) {
						RA(col, i + GridCount * (j - 1)) = -scale;
					}
					RA(col, col) += scale;
				}
				if (!Solid(i, j + 1)) {
					float scale = 1;// / SRV(Mass, i, j + 1).y;
					if (!Air(i, j + 1)) {
						RA(col, i + GridCount * (j + 1)) = -scale;
					}
					RA(col, col) += scale;
				}
			}

			// solve A*x=B
			{
				#pragma omp parallel for
				for (int i = 0; i < GridCount; i++) for (int j = 0; j < GridCount; j++) if (Air(i, j)) 
				{
					X[i][j] = 0;
				}

				// Conjugate gradient method
				if (1) {
					double(*r)[GridCount] = new double[GridCount][GridCount];
					double(*p)[GridCount] = new double[GridCount][GridCount];
					double(*Ap)[GridCount] = new double[GridCount][GridCount];
					memset(r, 0, sizeof(double) * GridCount * GridCount);
					memset(p, 0, sizeof(double)* GridCount* GridCount);
					memset(Ap, 0, sizeof(double) * GridCount * GridCount);
					#pragma omp parallel for
					for (int id = 0; id < index_count; id++) {
						int col = index[id];
						int vi = col % GridCount, vj = col / GridCount;
						p[vi][vj] = r[vi][vj] = B[col] - (
								RA(col, col) * RX( vi, vj) +
								RA(col, (vi + 1) + vj * GridCount) * RX( vi + 1, vj) +
								RA(col, (vi - 1) + vj * GridCount) * RX( vi - 1, vj) +
								RA(col, vi + (vj + 1) * GridCount) * RX( vi, vj + 1) +
								RA(col, vi + (vj - 1) * GridCount) * RX( vi, vj - 1)
								);
					}
					double r_r = 0;
					for (int id = 0; id < index_count; id++) {
						int col = index[id];
						int vi = col % GridCount, vj = col / GridCount;
						r_r += r[vi][vj] * r[vi][vj];
					}
					int total_iter = 10000;
					for (int loop = 0; loop < total_iter; loop++) {
						#pragma omp parallel for
						for (int i = 0; i < index_count; i++)
						{
							int col = index[i];
							int vi = col % GridCount, vj = col / GridCount;
							Ap[vi][vj] =
								RA(col, col) * SRD(p, vi, vj) +
								RA(col, (vi + 1) + vj * GridCount) * SRD(p, vi + 1, vj) +
								RA(col, (vi - 1) + vj * GridCount) * SRD(p, vi - 1, vj) +
								RA(col, vi + (vj + 1) * GridCount) * SRD(p, vi, vj + 1) +
								RA(col, vi + (vj - 1) * GridCount) * SRD(p, vi, vj - 1);
						}
						double p_A_p = 0;
						for (int id = 0; id < index_count; id++) {
							int col = index[id];
							int vi = col % GridCount, vj = col / GridCount;
							p_A_p += p[vi][vj] * Ap[vi][vj];
						}
						double a = p_A_p == 0 ? 0 : r_r / p_A_p;
						#pragma omp parallel for
						for (int id = 0; id < index_count; id++) {
							int col = index[id];
							int vi = col % GridCount, vj = col / GridCount;
							X[vi][vj] += a * p[vi][vj];
							r[vi][vj] += -a * Ap[vi][vj];
						}
						double next_r_r = 0;
						for (int id = 0; id < index_count; id++) {
							int col = index[id];
							int vi = col % GridCount, vj = col / GridCount;
							next_r_r += r[vi][vj] * r[vi][vj];
						}
						double beta = r_r == 0 ? 0 : next_r_r / r_r;
						r_r = next_r_r;
						if (r_r < 1e-5) {
							printf("early out with %d iter  ", loop);
							break;
						}
						#pragma omp parallel for
						for (int id = 0; id < index_count; id++) {
							int col = index[id];
							int vi = col % GridCount, vj = col / GridCount;
							p[vi][vj] = r[vi][vj] + beta * p[vi][vj];
						}
					}
					printf("error: %f x10^-5\n", r_r * 1e5);
					delete[] r;
					delete[] p;
					delete[] Ap;
				}

				// Preconditioning conjugate gradient method
				if (0) {
					auto Preconditioner = [](double(*a)[GridCount], double(*b)[GridCount]) {
						if (1) {
							//#pragma omp parallel for
							//for (int id = 0; id < index_count; id++) {
							//	int col = index[id];
							//	int vi = col % GridCount, vj = col / GridCount;

							//	double res = 0;
							//	if (vi < GridCount - 1)
							//		res = a[vi][vj] - 1.0 / RA(vi + 1 + vj * GridCount, vi + 1 + vj * GridCount) 
							//								* RA(col, vi + 1 + vj * GridCount) * SRD(a, vi + 1, vj);
							//	if (vj < GridCount - 1)
							//		res = a[vi][vj] - 1.0 / RA(vi + (vj + 1) * GridCount, vi + (vj + 1) * GridCount)
							//								* RA(col, vi + (vj + 1) * GridCount) * SRD(a, vi, vj + 1);

							//	temp[vi][vj] = res;// a[vi][vj] - res / RA(col, col);
							//}
							#pragma omp parallel for
							for (int id = 0; id < index_count; id++) {
								int col = index[id];
								int vi = col % GridCount, vj = col / GridCount;

								double res = 0;
								if (vi >= 1)
									res += RA(col, vi - 1 + vj * GridCount) * SRD(a, vi - 1, vj);
								if (vj >= 1)
									res += RA(col, vi + (vj - 1) * GridCount) * SRD(a, vi, vj - 1);

								b[vi][vj] = a[vi][vj] = a[vi][vj] - res / RA(col, col);
							}
						}
						// Incomplete Poisson
						else if (1) {
							#pragma omp parallel for
							for (int id = 0; id < index_count; id++) {
								int col = index[id];
								int vi = col % GridCount, vj = col / GridCount;

								double res = 0;
								if (vi < GridCount - 1)
									res += RA(col, vi + 1 + vj * GridCount) * SRD(a, vi + 1, vj);
								if (vj < GridCount - 1)
									res += RA(col, vi + (vj + 1) * GridCount) * SRD(a, vi, vj + 1);

								temp[vi][vj] = a[vi][vj] - res / RA(col, col);
							}
							#pragma omp parallel for
							for (int id = 0; id < index_count; id++) {
								int col = index[id];
								int vi = col % GridCount, vj = col / GridCount;

								double res = 0;
								if (vi >= 1)
									res += RA(col, vi - 1 + vj * GridCount) * SRD(temp, vi - 1, vj);
								if (vj >= 1)
									res += RA(col, vi + (vj - 1) * GridCount) * SRD(temp, vi, vj - 1);

								b[vi][vj] = temp[vi][vj] = temp[vi][vj] - res / RA(col, col);
							}
						}
						// None
						else {
							memcpy(b, a, sizeof(double) * GridCount * GridCount); return;
						}
					};

					double(*r)[GridCount] = new double[GridCount][GridCount];
					double(*z)[GridCount] = new double[GridCount][GridCount];
					double(*s)[GridCount] = new double[GridCount][GridCount];
					memset(r, 0, sizeof(double) * GridCount * GridCount);
					memset(z, 0, sizeof(double) * GridCount * GridCount);
					memset(s, 0, sizeof(double) * GridCount* GridCount);
					memset(temp, 0, sizeof(double) * GridCount * GridCount);
					#pragma omp parallel for
					for (int id = 0; id < index_count; id++) {
						int col = index[id];
						int vi = col % GridCount, vj = col / GridCount;
						r[vi][vj] = B[col] - (
							RA(col, col) * RX(vi, vj) +
							RA(col, (vi + 1) + vj * GridCount) * RX(vi + 1, vj) +
							RA(col, (vi - 1) + vj * GridCount) * RX(vi - 1, vj) +
							RA(col, vi + (vj + 1) * GridCount) * RX(vi, vj + 1) +
							RA(col, vi + (vj - 1) * GridCount) * RX(vi, vj - 1)
							);
					}
					Preconditioner(r, z);
					memcpy(s, z, sizeof(double) * GridCount * GridCount);
					double theta = 0;
					for (int id = 0; id < index_count; id++) {
						int col = index[id];
						int vi = col % GridCount, vj = col / GridCount;
						theta += z[vi][vj] * r[vi][vj];
					}
					int total_iter = 1000000;
					for (int loop = 0; loop < total_iter; loop++) {
						#pragma omp parallel for
						for (int i = 0; i < index_count; i++)
						{
							int col = index[i];
							int vi = col % GridCount, vj = col / GridCount;
							z[vi][vj] =
								RA(col, col) * SRD(s, vi, vj) +
								RA(col, (vi + 1) + vj * GridCount) * SRD(s, vi + 1, vj) +
								RA(col, (vi - 1) + vj * GridCount) * SRD(s, vi - 1, vj) +
								RA(col, vi + (vj + 1) * GridCount) * SRD(s, vi, vj + 1) +
								RA(col, vi + (vj - 1) * GridCount) * SRD(s, vi, vj - 1);
						}
						double zdots= 0;
						for (int id = 0; id < index_count; id++) {
							int col = index[id];
							int vi = col % GridCount, vj = col / GridCount;
							zdots += z[vi][vj] * s[vi][vj];
						}
						double alpha = zdots == 0 ? 0 : theta / zdots;
						#pragma omp parallel for
						for (int id = 0; id < index_count; id++) {
							int col = index[id];
							int vi = col % GridCount, vj = col / GridCount;
							X[vi][vj] += alpha * s[vi][vj];
							r[vi][vj] += -alpha * z[vi][vj];
						}
						Preconditioner(r, z);
						double new_theta = 0;
						for (int id = 0; id < index_count; id++) {
							int col = index[id];
							int vi = col % GridCount, vj = col / GridCount;
							new_theta += z[vi][vj] * r[vi][vj];
						}
						double beta = theta == 0 ? 0 : new_theta / theta;
						theta = new_theta;
						printf("error: %f x10^-5\n", theta * 1e5);
						if (theta < 1e-15) {
							printf("early out with %d iter  ", loop);
							break;
						}
						#pragma omp parallel for
						for (int id = 0; id < index_count; id++) {
							int col = index[id];
							int vi = col % GridCount, vj = col / GridCount;
							s[vi][vj] = z[vi][vj] + beta * s[vi][vj];
						}
					}
					printf("error: %f x10^-5\n", theta * 1e5);
					delete[] r;
					delete[] z;
					delete[] s;
				}

				// Toy iteration Method
				if (0)
				{
					int total_iter = GridCount;
					for (int loop = 0; loop < total_iter; loop++)
					{
						#pragma omp parallel for
						for (int id = 0; id < index_count; id++) {
							int col = index[id];
							int vi = col % GridCount, vj = col / GridCount;
							double k =
								RA(col, (vi + 1) + vj * GridCount) * RX( vi + 1, vj) +
								RA(col, (vi - 1) + vj * GridCount) * RX( vi - 1, vj) +
								RA(col, vi + (vj + 1) * GridCount) * RX( vi, vj + 1) +
								RA(col, vi + (vj - 1) * GridCount) * RX( vi, vj - 1);
							X[vi][vj] = (B[col] - k) / RA(col, col);
						}
					}
				}

				//// GPU solver
				//if (0)
				//	Run(GridCount, index, index_count, A, B, Pressure);

				// Update velocity use pressure
				#pragma omp parallel for
				for (int i = 0; i <= GridCount; i++) for (int j = 0; j <= GridCount; j++)
				{
					Vector2 deltaV = Vector2(RX( i - 1, j) - RX( i, j), RX( i, j - 1) - RX( i, j));
					if (Solid(i, j)) deltaV = Vector2(0);
					else {
						if (Solid(i - 1, j)) deltaV.x = 0;
						if (Solid(i, j - 1)) deltaV.y = 0;
					}
					//deltaV /= Mass[i][j];
					Velocity[i][j] += deltaV;
				}
				#pragma omp parallel for
				for (int i = 0; i <= GridCount; i++) for (int j = 0; j <= GridCount; j++)
				{
					Vector2 v = Velocity[i][j];
					Vector2 m = Mass[i][j];
					if (m.x == 0 || m.y == 0) {
						Vector2 vx = Vector2(0);
						Vector2 vy = Vector2(0);
						for (int oi = -1; oi <= 1; oi++) for (int oj = -1; oj <= 1; oj++)
						{
							Vector2 mass = SRV(Mass, i + oi, j + oj);
							Vector2 vel = SRV(Velocity_, i + oi, j + oj);
							vx += mass.x != 0 ? Vector2(vel.x, 1) : Vector2(0);
							vy += mass.y != 0 ? Vector2(vel.y, 1) : Vector2(0);
						}
						v.x = m.x == 0 ? (vx.y != 0 ? vx.x / vx.y : 0) : v.x;
						v.y = m.y == 0 ? (vy.y != 0 ? vy.x / vy.y : 0) : v.y;
					}
					Velocity[i][j] = v;
				}
			}
		}

		// G2P
		{
			#pragma omp parallel for
			for (int i = 0; i < Particles.size(); i++)
			{
				Particle& p = Particles[i];

				//Vector2 k1 = CenterSpeed(Velocity, p.x.cast<int>()) + (p.v - CenterSpeed(Velocity_, p.x.cast<int>())) * 0.9f;
				//Vector2 x = p.x + k1 * 0.5f * deltaT;
				//Vector2 k2 = CenterSpeed(Velocity, x.cast<int>()) + (p.v - CenterSpeed(Velocity_, x.cast<int>())) * 0.9f;
				//x = p.x + k2 * 0.75f * deltaT;
				//Vector2 k3 = CenterSpeed(Velocity, x.cast<int>()) + (p.v - CenterSpeed(Velocity_, x.cast<int>())) * 0.9f;
				//p.v = (k1* (2.f / 9) + k2 * (3.f / 9) + k3 * (4.f / 9));

				Vector2 k1;
				k1.x = EdgeSpeedXX(Velocity, p.x - Vector2(0, 0.5)) + (p.v.x - EdgeSpeedXX(Velocity_, p.x - Vector2(0, 0.5))) * 0.95;
				k1.y = EdgeSpeedYY(Velocity, p.x - Vector2(0.5, 0)) + (p.v.y - EdgeSpeedYY(Velocity_, p.x - Vector2(0.5, 0))) * 0.95;
				Vector2 x = p.x + k1 * 0.5f * deltaT;
				Vector2 k2;
				k2.x = EdgeSpeedXX(Velocity, x - Vector2(0, 0.5)) + (p.v.x - EdgeSpeedXX(Velocity_, x - Vector2(0, 0.5))) * 0.95;
				k2.y = EdgeSpeedYY(Velocity, x - Vector2(0.5, 0)) + (p.v.y - EdgeSpeedYY(Velocity_, x - Vector2(0.5, 0))) * 0.95;
				x = p.x + k2 * 0.75f * deltaT;
				Vector2 k3;
				k3.x = EdgeSpeedXX(Velocity, x - Vector2(0, 0.5)) + (p.v.x - EdgeSpeedXX(Velocity_, x - Vector2(0, 0.5))) * 0.95;
				k3.y = EdgeSpeedYY(Velocity, x - Vector2(0.5, 0)) + (p.v.y - EdgeSpeedYY(Velocity_, x - Vector2(0.5, 0))) * 0.95;
				p.v = k1;// (k1* (2.f / 9) + k2 * (3.f / 9) + k3 * (4.f / 9));

				p.x += p.v * deltaT;

				float boundary = 0.04, bx = p.x.x / GridCount, by = p.x.y / GridCount;
				bx = min((int((1 - boundary) / dx) + 0.5f) * dx, max(bx, (int(boundary / dx) + 0.5f) * dx));
				by = min((int((1 - boundary) / dx) + 0.5f) * dx, max(by, (int(boundary / dx) + 0.5f) * dx));
				p.x = Vector2(bx, by) * (float)GridCount;
			}
		}
	}

	int main() {

		//original::main();

		GUI gui("FLIP", GridCount * 2, GridCount * 2);

		Init();

		auto& canvas = gui.get_canvas(); int f = 0;
		for (int i = 0;; i++) {                              //              Main Loop
			Simulate(dt);
			if (i % int(frame_dt / dt) == 0) {                 //        Visualize frame
				canvas.clear(0x112F41);                          //       Clear background
				#pragma omp parallel for
				for (int i = 0; i < Particles.size(); i++)
				{
					Particle p = Particles[i];
					Vector2i ij = p.x.cast<int>();
					Vector2 xy = Velocity[ij.x][ij.y];
					float lx = xy.x;
					float dy = xy.y;
					float rx = SRV(Velocity, ij.x + 1, ij.y).x;
					float uy = SRV(Velocity, ij.x, ij.y + 1).y;
					float value = (lx - rx + dy - uy) * 100;// sqrt(length(p.v) / GridCount / 30);//  pow(Pressure[ij.x][ij.y] / 50000, 1 / 2.2);// length(p.v) / 1000; //xy.y / 1000; 
					int c = max(0.0f, min(1.0f, value)) * 255;
					int c2 = max(0.0f, min(1.0f, -value)) * 255;
					canvas.circle(p.x / Vector2(GridCount)).radius(1).color((c << 16) + (c2 << 8));// +255);
				}
				//#pragma omp parallel for
				//for (int i = 0; i < GridCount; i++) for (int j = 0; j < GridCount; j++)
				//{
				//	Vector2i ij = Vector2i(i, j);

				//	Vector2 xy = Velocity[ij.x][ij.y];
				//	float lx = xy.x;
				//	float dy = xy.y;
				//	float rx = SRV(Velocity, ij.x + 1, ij.y).x;
				//	float uy = SRV(Velocity, ij.x, ij.y + 1).y;

				//	float value = (lx - rx + dy - uy);
				//	int c = int(max(0.0f, min(1.0f, value)) * 255) << 16;
				//	int c2 = int(max(0.0f, min(1.0f, -value)) * 255) << 8;
				//	canvas.circle(Vector2((i + 0.5) / GridCount, (j + 0.5) / GridCount)).radius(1).color(c+c2);// (!Solid(i, j) && !Air(i, j)) * 255);
				//}
				//#pragma omp parallel for
				//for (int i = 0; i < GridCount; i++) for (int j = 0; j < GridCount; j++)
				//{
				//	float value = Air(i, j);
				//	int c = max(0.0f, min(1.0f, value)) * 255;
				//	canvas.circle(Vector2((i + 0.5) / GridCount, (j + 0.5) / GridCount)).radius(2).color((c << 16) + (c << 8) + c);//Particles
				//}
				//#pragma omp parallel for
				//for (int i = 0; i <= GridCount; i++) for (int j = 0; j <= GridCount; j++)
				//{
				//	Vector2 xy = Velocity[i][j];
				//	float value = xy.x / 1000;
				//	int c = max(0.0f, min(1.0f, value)) * 255;
				//	int c2 = max(0.0f, min(1.0f, -value)) * 255;// c;
				//	canvas.circle(Vector2(float(i) / GridCount, (j + 0.5) / GridCount)).radius(1).color((c << 16) + (c2 << 8));//Particles
				//	value = xy.y / 1000;// ; ;// -3.5; //
				//	c = max(0.0f, min(1.0f, value)) * 255;
				//	c2 = max(0.0f, min(1.0f, -value)) * 255;// c;
				//	canvas.circle(Vector2((i + 0.5) / GridCount, float(j) / GridCount)).radius(1).color((c << 16) + (c2 << 8));//Particles
				//}
				canvas.rect(Vector2(0.04), Vector2(0.96)).radius(2).color(0x4FB99F).close();// Box
				gui.update();
			}
		}
	}
}


int main() {
	FLIP::main();
}