// flann-test.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <flann/flann.hpp>
#include <flann/io/hdf5.h>
#include <stdio.h>

using namespace flann;

int main(int argc, char** argv)
{
	int nn = 3;

	Matrix<float> dataset;
	Matrix<float> query;
	load_from_file(dataset, "dataset.hdf5", "dataset");
	load_from_file(query, "dataset.hdf5", "query");

	Matrix<int> indices(new int[query.rows*nn], query.rows, nn);
	Matrix<float> dists(new float[query.rows*nn], query.rows, nn);

	// construct an randomized kd-tree index using 4 kd-trees
	Index<L2<float> > index(dataset, flann::KDTreeIndexParams(4));
	index.buildIndex();

	// do a knn search, using 128 checks
	index.knnSearch(query, indices, dists, nn, flann::SearchParams(128));

	for (int i = indices.rows; i < indices.rows; i++)
	{
		for (int j = 0; j < indices.cols; j++)
		{
			std::cout << indices[i][j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << "index size: " << index.size() << std::endl;
	std::cout << std::endl;

	flann::save_to_file(indices, "result.hdf5", "result");

	delete[] dataset.ptr();
	delete[] query.ptr();
	delete[] indices.ptr();
	delete[] dists.ptr();

	return 0;
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
