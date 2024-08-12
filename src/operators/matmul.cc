#include "operators/matmul.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        if (inputs.size() != 2)
            return std::nullopt;
        Shape shapeA = inputs[0]->getDims();
        Shape shapeB = inputs[1]->getDims();

        bool transA = this->getTransA();
        bool transB = this->getTransB();
        if (transA)
        {
            std::swap(shapeA[shapeA.size() - 2], shapeA[shapeA.size() - 1]);
        }
        if (transB)
        {
            std::swap(shapeB[shapeB.size() - 2], shapeB[shapeB.size() - 1]);
        }
        if (shapeA[shapeA.size() - 1] != shapeB[shapeB.size() - 2])
        {
            return std::nullopt;
        }
        Shape shapeC = shapeA;
        shapeC[shapeC.size() - 1] = shapeB[shapeB.size() - 1];
        return std::vector<Shape>{shapeC};
    }
} // namespace infini