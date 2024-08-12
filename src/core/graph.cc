#include "core/graph.h"
#include <algorithm>
#include <numeric>
#include <queue>
#include "operators/matmul.h"
#include "operators/transpose.h"
namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    void GraphObj::optimize()
    {
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来实现指定的图优化规则
        // 图优化规则如下：
        // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
        // 2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
        // =================================== 作业 ===================================
        // 去除冗余的算子
        for (size_t i = 0; i < ops.size(); ++i)
        {
            Operator op = ops[i];
            if (op->getOpType() == OpType::Transpose)
            {
                Tensor tensor = op->getOutput();
                if (!tensor)
                    continue;
                auto targets = tensor->getTargets();
                if (targets.empty())
                    continue;
                Operator op_next = targets[0];
                if (op_next->getOpType() == OpType::Transpose)
                {
                    TransposeObj *op1 = as<TransposeObj>(op).get();
                    TransposeObj *op2 = as<TransposeObj>(op_next).get();
                    auto op1_permute = op1->getPermute();
                    auto op2_permute = op2->getPermute();
                    if (op1_permute.size() != op2_permute.size())
                        continue;
                    bool flag = true;
                    for (int j = 0; j < (int)op1_permute.size(); j++)
                    {
                        if (op1_permute[op2_permute[j]] != j)
                        {
                            flag = false;
                            continue;
                        }
                    }
                    if (!flag)
                        continue;
                    Tensor a = op->getInputs()[0];
                    Tensor b = op->getOutput();
                    Tensor c = op_next->getOutput();
                    auto op3 = c->getTargets()[0];
                    auto d = op3->getInputs()[1];
                    op3->replaceInput(op3->getInputs()[0], a);
                    a->removeTarget(op);
                    a->addTarget(op3);
                    a->setSource(nullptr);
                    removeOperator(op);
                    removeOperator(op_next);
                    removeTensor(b);
                    removeTensor(c);

                    op3->removePredecessors(op_next);
                    if (a->getSource())
                    {
                        op3->addPredecessors(a->getSource());
                        a->getSource()->addSuccessors(op3);
                    }
                }
            }
        }
        // 合并算子
        for (size_t i = 0; i < ops.size(); ++i)
        {
            Operator op = ops[i];
            if (op->getOpType() == OpType::MatMul)
            {
                TensorVec tensorvec = op->getInputs();
                int nu_i = 0;
                for (Tensor tensor : tensorvec)
                {
                    nu_i++;
                    if (tensor->getSource())
                    {
                        Operator op1 = tensor->getSource();
                        if (op1->getOpType() == OpType::Transpose)
                        {
                            TransposeObj *transpose_op = as<TransposeObj>(op1).get();
                            Shape t_perm = transpose_op->getPermute();
                            bool flag = true;
                            for (int j = 0; j < (int)t_perm.size() - 2; j++)
                            {
                                if (j < ((int)t_perm.size() - 2) && t_perm[j] != j)
                                {
                                    flag = false;
                                    continue;
                                }
                            }
                            if (t_perm[t_perm.size() - 2] != (int)t_perm.size() - 1 && t_perm[t_perm.size() - 1] != (int)t_perm.size() - 2)
                                flag = false;
                            if (flag == false)
                                continue;
                            MatmulObj *mul_op = as<MatmulObj>(op).get();
                            Tensor tensor1; // 6
                            if (nu_i == 1)
                            {
                                mul_op->setTransA(true);
                                tensor1 = mul_op->getInputs(0);
                            }
                            else
                            {
                                mul_op->setTransB(true);
                                tensor1 = mul_op->getInputs(1);
                            }
                            Operator operator1 = tensor1->getSource();
                            Tensor tensor2 = operator1->getInputs(0); // 3
                            mul_op->replaceInput(tensor1, tensor2);
                            tensor2->removeTarget(operator1);
                            tensor2->addTarget(op);
                            removeOperator(operator1);
                            removeTensor(tensor1);

                            op->removePredecessors(operator1);
                            if (tensor2->getSource())
                            {
                                op->addPredecessors(tensor2->getSource());
                                tensor2->getSource()->addSuccessors(op);
                            }
                        }
                    }
                }
            }
        }
    }

    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        // =================================== 作业 ===================================
        // TODO：利用 allocator 给计算图分配内存
        // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor 绑定内存
        // =================================== 作业 ===================================
        vector<size_t> vec_offset;
        for (auto tensor : tensors)
        {
            size_t size = tensor->getBytes();
            size_t offset = allocator.alloc(size);
            vec_offset.push_back(offset);
        }
        auto ve = vec_offset.begin();
        void *basePtr = allocator.getPtr();
        for (auto tensor : tensors)
        {
            char *charPtr = reinterpret_cast<char *>(basePtr) + *ve;
            void *ptr = charPtr;
            Blob blob = make_ref<BlobObj>(runtime, ptr);
            tensor->setDataBlob(blob);
            ve++;
        }
        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini