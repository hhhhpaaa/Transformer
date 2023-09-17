``` python
class Attention_GRU(nn.Module):  
  
    def __init__(self, tensor_size, num_heads, DEVICE, class_num, seq_len, input_size):  
        super().__init__()  
        self.tensor_size = tensor_size  
        self.seq_len = seq_len  
        self.input_size_ = input_size  
        self.gru1 = nn.GRU(batch_first=True, input_size=self.input_size_, hidden_size=self.tensor_size)  
        self.gru2 = nn.GRU(batch_first=True, input_size=self.tensor_size, hidden_size=self.tensor_size)  
        self.self_attention = Self_Attention(tensor_size, num_heads, DEVICE)  
        self.layer_norm = nn.LayerNorm(normalized_shape=self.tensor_size)  
        self.linear1 = nn.Linear(in_features=self.tensor_size * seq_len, out_features=self.tensor_size)  
        self.linear2 = nn.Linear(in_features=self.tensor_size, out_features=class_num)  
  
    def forward(self, tensor_input):  
  
        output, _ = self.gru1(tensor_input.reshape(-1, self.seq_len, self.input_size_))  
        output_all, output_end = self.gru2(output)  
        attention_output = self.self_attention(output_end.transpose(0, 1), output_all, output_all)  
        output = self.layer_norm(attention_output + output_all)  
        output = F.relu(self.linear1(output.reshape(-1, self.seq_len * self.tensor_size)))  
        output = self.linear2(output)  
  
        return output  
  
  
class Attention_LSTM(nn.Module):  
  
    def __init__(self, tensor_size, num_heads, DEVICE, class_num, seq_len, input_size):  
        super().__init__()  
        self.tensor_size = tensor_size  
        self.seq_len = seq_len  
        self.input_size_ = input_size  
        self.gru1 = nn.LSTM(batch_first=True, input_size=self.input_size_, hidden_size=self.tensor_size)  
        self.gru2 = nn.LSTM(batch_first=True, input_size=self.tensor_size, hidden_size=self.tensor_size)  
        self.self_attention = Self_Attention(tensor_size, num_heads, DEVICE)  
        self.layer_norm = nn.LayerNorm(normalized_shape=self.tensor_size)  
        self.linear1 = nn.Linear(in_features=self.tensor_size * seq_len, out_features=self.tensor_size)  
        self.linear2 = nn.Linear(in_features=self.tensor_size, out_features=class_num)  
  
    def forward(self, tensor_input):  
  
        output, _ = self.gru1(tensor_input.reshape(-1, self.seq_len, self.input_size_))  
        output_all, (output_end, _) = self.gru2(output)  
        attention_output = self.self_attention(output_end.transpose(0, 1), output_all, output_all)  
        output = self.layer_norm(attention_output + output_all)  
        output = F.relu(self.linear1(output.reshape(-1, self.seq_len * self.tensor_size)))  
        output = self.linear2(output)  
  
        return output  
  
  
class Self_Attention(nn.Module):  
  
    def __init__(self, tensor_size, num_heads, DEVICE):  
        super().__init__()  
        self.tensor_size = tensor_size  
        self.num_heads = num_heads  
        self.DEVICE = DEVICE  
        assert self.tensor_size % self.num_heads == 0, "tensor_size % num_heads not zero"  
        self.depth = self.tensor_size // self.num_heads  
        self.q = nn.Linear(in_features=self.tensor_size, out_features=self.tensor_size)  
        self.k = nn.Linear(in_features=self.tensor_size, out_features=self.tensor_size)  
        self.v = nn.Linear(in_features=self.tensor_size, out_features=self.tensor_size)  
        self.output_ = nn.Linear(in_features=self.tensor_size, out_features=self.tensor_size)  
        self.softmax = nn.Softmax(dim=-1)  
  
    def split_heads(self, tensor_input, batch_size):  
  
        return tensor_input.reshape(batch_size, self.num_heads, -1, self.depth)  
  
    def forward(self, q_input, k_input, v_input):  
  
        batch_size = q_input.shape[0]  
  
        q = self.split_heads(self.q(q_input), batch_size)  
        k = self.split_heads(self.k(k_input), batch_size)  
        v = self.split_heads(self.v(v_input), batch_size)  
  
        mul_weight = torch.matmul(q, k.transpose(3, 2)) / torch.sqrt(torch.Tensor([q.shape[-1]])).to(self.DEVICE)  
        attention_weight = self.softmax(mul_weight)  
        output_attention = torch.matmul(attention_weight, v).reshape(batch_size, -1, self.tensor_size)  
        output = self.output_(output_attention)  
  
        return output
```

### test

``` python
model_ = Attention_LSTM(tensor_size=64, num_heads=4, DEVICE='cuda:0', class_num=2, seq_len=42, input_size=1).to('cuda:0')  
# model_ = NetWork(embedding_len=1, units=64, num_class=2)  
print(model_(torch.rand(64, 42).to('cuda:0')).shape)
```
