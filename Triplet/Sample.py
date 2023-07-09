import torch
import torch.nn as nn
import torch.nn.functional as F

start_logits = torch.tensor([[-0.2554, -0.0857],
                             [-0.3506, 0.3321],
                             [-0.4001, -0.6767],
                             [-0.4544, 0.3110]])

end_logits = torch.tensor([[0.1001, -0.2002],
                           [-0.1503, 0.2504],
                           [0.3005, -0.4006],
                           [-0.1234, 0.5678]])

start_positions = torch.tensor([0, 1, 0, 0])
end_positions = torch.tensor([0, 0, 1, 0])

criterion = nn.CrossEntropyLoss()

def spanloss(start_logits,end_logits,start_positions,end_positions):

    # 得到start_probs和end_probs
    start_probs = F.softmax(start_logits, dim=1)
    end_probs = F.softmax(end_logits, dim=1)

    # 计算标签的分布得到target_probs
    target_probs = torch.zeros_like(start_probs)
    start_positions = start_positions.unsqueeze(-1)  # Add a new dimension at the end
    target_probs.scatter_(1, start_positions, 1)
    end_positions = end_positions.unsqueeze(-1)  # Add a new dimension at the end
    target_probs.scatter_(1, end_positions, 1)

    # 损失函数计算start_probs和end_probs与target_probs之间的差值
    loss_fn = nn.CrossEntropyLoss()
    start_loss = loss_fn(start_probs, target_probs)
    end_loss = loss_fn(end_probs, target_probs)
    sopan_loss = start_loss + end_loss
    return sopan_loss



start_loss = criterion(start_logits, start_positions)
end_loss = criterion(end_logits, end_positions)
sopan_loss=spanloss(start_logits,end_logits,start_positions,end_positions)
total_loss = start_loss + end_loss + sopan_loss

print("Total Loss:", total_loss.item())



# PATH = "model_weights.pth"
# torch.save(model.state_dict(), PATH)
#
# # 在所有的epoch训练完成后
# if epoch == num_epochs:
# # 保存模型的权重
# torch.save(model.state_dict(), PATH)
# print("模型权重已保存")
#
# # 加载模型权重
# model.load_state_dict(torch.load(PATH))
