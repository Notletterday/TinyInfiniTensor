#include "operators/concat.h"
#include "utils/operator_utils.h"

namespace infini
{
    ConcatObj::ConcatObj(GraphObj *graph, TensorVec inputs, Tensor output, int _dim)
        : OperatorObj(OpType::Concat, inputs, {output})
    {
        int rank = inputs[0]->getRank();
        dim = get_real_axis(_dim, rank);
        IT_ASSERT(checkValid(graph));
    }

    optional<vector<Shape>> ConcatObj::inferShape(const TensorVec &inputs)
    {
        Shape dims = inputs[0]->getDims();
        auto rank = inputs[0]->getRank();
        // =================================== 作业 ===================================
        // TODO：修改 dims，返回正确的 concat 后的 shape
        // REF: https://onnx.ai/onnx/operators/onnx__Concat.html#concat-13
        // =================================== 作业 ===================================
        if (inputs.empty())
            return std::nullopt;
        for (auto input : inputs)
            if (input->getRank() != rank)
                return std::nullopt;
        std::vector<int> outputDims(rank, 0);
        for (auto input : inputs)
            for (int i = 0; i < (int)rank; i++)
                if (i == dim)
                    outputDims[i] += input->getDims()[i];
                else if (outputDims[i] == 0)
                    outputDims[i] = input->getDims()[i];
                else if (outputDims[i] != input->getDims()[i])
                    return std::nullopt;
        return {{outputDims}};
    }

    std::string ConcatObj::toString() const
    {
        std::ostringstream os;
        os << "Concat[" << getGuid() << "]";
        os << "(";
        for (auto input : inputs)
            os << vecToString(input->getDims()) << ",";
        os << "dim=" << dim << ",";
        os << "input=";
        for (auto input : inputs)
            os << input->getGuid() << ",";
        os << "output=" << outputs[0]->getGuid() << ")";
        return os.str();
    }

} // namespace infini
