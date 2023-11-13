# gcc ./src/test_nabla_scalar.c  -o test -lm
g++ -I ~/Toolbox/eigen-3.4.0/ ./src/test_nabla.cpp -o test -lm
./test
