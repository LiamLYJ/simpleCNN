#include <iostream>
#include <string>
#include <vector>
#include "./npy.h"

using namespace std;

template<typename T>
int test_load(string fn, vector<T> data)
{
    vector<unsigned long> shape;

    npy::LoadArrayFromNumpy(fn, shape, data);

    cout << "shape: ";
    for (size_t i = 0; i < shape.size(); i++)
        cout << shape[i] << ", ";
    cout << endl;
//    cout << "data: ";
//    for (size_t i = 0; i < data.size(); i++)
//        cout << data[i] << ", ";
//    cout << endl;

    return 0;
}


int test_save(void)
{
    const long unsigned leshape[] = {2, 3};
    vector<double> data{1, 2, 3, 4, 5, 6};
    npy::SaveArrayAsNumpy("data/out.npy", false, 2, leshape, data);

    const long unsigned leshape2[] = {6};
    npy::SaveArrayAsNumpy("data/out2.npy", false, 1, leshape2, data);

    return 0;
}

int main(int argc, char **argv)
{
    string fn = "./data/weight_01.npy";
    vector<float> data;
//    string fn1 = "./data/input.npy";
//    vector<uint8_t> data1;
    test_load(fn,data);
//    test_load(fn1,data1);

//    vector <int> a(10,1);
//    vector <int> b(10,2);
//    vector <int> c;
//    c.assign(a.begin(), a.end());
    
    return 0;
}


