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
    void fx(Tensor* x, Tensor* y);
    void vjp(Tensor* y, Tensor* x, MatrixXd grad_x);
}


class Add: public Operator
{
public:
    void fx(Tensor* x1, Tensor* x2, Tensor* y)
    {   
        strcpy(y->func_name, "add");
        y->data = x1->data + x2->data;
    }
    void vjp(Tensor* y, Tensor* x1, Tensor* x2, MatrixXd* grad_x1, MatrixXd* grad_x2)
    {
        *grad_x1 = y->grad;
        *grad_x2 = y->grad;
    }
}

class Mul: public Operator
{
public:
    void fx(Tensor* x1, Tensor* x2, Tensor* y)
    {   
        strcpy(y->func_name, "mul");
        *y = x1->data * x2->data;
    }
    void vjp(Tensor* y, Tensor* x1, Tensor* x2, MatrixXd* grad_x1, MatrixXd* grad_x2)
    {
        *grad_x1 = y->grad * x2->data;
        *grad_x2 = y->grad * x1->data;
    }
}