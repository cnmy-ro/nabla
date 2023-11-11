#include <string.h>
#include "Eigen/Dense"


using Eigen::MatrixXd;


class Tensor
{
private:
    MatrixXd data;
    MatrixXd grad;
    bool requires_grad;
    Tensor* prev[2];
    char func_name[10];

public:
    void Tensor(MatrixXd d, bool req_g=False)
    {
        data = d;
        requires_grad = req_g;
    }
    void backward(MatrixXd g)
    {
    }
}


class Operator
{
public:
    void fx(Tensor* x, Tensor* eval);
    void vjp(Tensor* eval, Tensor* x, MatrixXd grad_x);
}


class Add: public Operator
{
public:
    void fx(Tensor* x1, Tensor* x2, Tensor* eval)
    {   
        strcpy(eval->func_name, "add");
        eval->data = x1->data + x2->data;
    }
    void vjp(Tensor* eval, Tensor* x1, Tensor* x2, MatrixXd* grad_x1, MatrixXd* grad_x2)
    {
        *grad_x1 = eval->grad;
        *grad_x2 = eval->grad;
    }
}

class Mul: public Operator
{
public:
    void fx(Tensor* x1, Tensor* x2, Tensor* eval)
    {   
        strcpy(eval->func_name, "mul");
        *eval = x1->data * x2->data;
    }
    void vjp(Tensor* eval, Tensor* x1, Tensor* x2, MatrixXd* grad_x1, MatrixXd* grad_x2)
    {
        *grad_x1 = eval->grad * x2->data;
        *grad_x2 = eval->grad * x1->data;
    }
}