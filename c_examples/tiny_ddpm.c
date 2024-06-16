#include <stdio.h>
#include <stdbool.h>
#include "../c/nabla.h"
#include "../c/arrays.h"


// ---
// Scheduler

struct LinearScheduler {
    float* betas;
    float* sigmas;
    float* alphas;
    float* alpha_cumprods;
    
};
typedef struct LinearScheduler LinearScheduler;

void init_scheduler(LinearScheduler* scheduler, int num_diffusion_steps) {
}
void destroy_scheduler(LinearScheduler* scheduler) {
}


// ---
// Model

struct NoiseModel {
    Tensor params[8];
};
typedef struct NoiseModel NoiseModel;

void init_model(NoiseModel* model, int hidden_dim) {
}
void destroy_model(NoiseModel* model) {
}
void model_zero_grad(NoiseModel* model) {
}


// ---
// Optimizer

struct AdamOptimizer {
    float lr, beta_1, beta_2;
    int t;
    Tensor m_state[8];
    Tensor v_state[8];
};
typedef struct AdamOptimizer AdamOptimizer;

void init_optimizer(AdamOptimizer* opt, int lr) {
}
void destroy_optimizer(AdamOptimizer* opt) {
}
void optimizer_step(AdamOptimizer* opt, NoiseModel* model) {
}


// ---
// Utils

void sample_data(Tensor* batch) {
}

void criterion(Tensor* batch, Tensor* t_batch, NoiseModel* model, Tensor* loss) {
}


// ---
// Main

void main() {

	// Config
	int hidden_dim = 256;
	int num_diffusion_steps = 1000;
	float beta_t = 1e-4;
	float lr = 1e-4;
	int batch_size = 128;
	int num_train_iters = 5000;

	// Scheduler
	LinearScheduler scheduler;
	init_scheduler(&scheduler, num_diffusion_steps);

	// Model
	NoiseModel model;
	init_model(&model, hidden_dim);

	// Optimizer
	AdamOptimizer opt;
	init_optimizer(&opt, lr);

	// Working tensors and arrays
	Tensor batch, t_batch, loss;
	malloc_tensor(&batch, 1, batch_size, true);
	malloc_tensor(&t_batch, 1, batch_size, true);
	malloc_tensor(&loss, 1, 1, true);
	Array init_grad;
	malloc_array(&init_grad, 1, 1);
	init_array_value(&init_grad, 1);

	// Training loop
	for (int it=1; it<=num_train_iters; it++) {
		sample_data(&batch);
        init_tensor_randint(&t_batch, 0, num_diffusion_steps);
        criterion(&batch, &t_batch, &model, &loss);
        backward(&loss, &init_grad);
        optimizer_step(&opt, &model);
        model_zero_grad(&model);

        if (it % 10 == 0)
        	print_tensor(&loss);
	}

	// Cleanup
	free_array(&init_grad);
	free_tensor(&loss);
	free_tensor(&t_batch);
	free_tensor(&batch);	
	destroy_optimizer(&opt);
	destroy_model(&model);
	destroy_scheduler(&scheduler);
}