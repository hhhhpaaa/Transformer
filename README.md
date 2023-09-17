# Transformer代码复现（Pytorch版本，不带Mask）

## Self_Attention

``` python
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

## Encoder_layer

``` python
class Encoder_layer(nn.Module):

    def __init__(self, tensor_size, num_heads, feed_forward_size, DEVICE):
        super().__init__()
        self.tensor_size = tensor_size
        self.DEVICE = DEVICE
        self.self_attention = Self_Attention(tensor_size, num_heads, DEVICE)
        self.droput1 = nn.Dropout()
        self.layer_norm1 = nn.LayerNorm(normalized_shape=self.tensor_size)
        self.droput2 = nn.Dropout()
        self.layer_norm2 = nn.LayerNorm(normalized_shape=self.tensor_size)
        self.feed_forward = nn.Sequential(nn.Linear(in_features=self.tensor_size, out_features=feed_forward_size),
                                          nn.ReLU(),
                                          nn.Linear(in_features=feed_forward_size, out_features=self.tensor_size))

    def forward(self, tensor_input):

        attention_output = self.self_attention(tensor_input, tensor_input, tensor_input)
        attention_output = self.droput1(attention_output)
        output = self.layer_norm1(tensor_input + attention_output)

        feed_forward_output = self.feed_forward(output)
        feed_forward_output = self.droput2(feed_forward_output)
        output = self.layer_norm2(output + feed_forward_output)

        return output
```

## Decoder_layer

``` python
class Decoder_layer(nn.Module):

    def __init__(self, tensor_size, num_heads, feed_forward_size, DEVICE):
        super().__init__()
        self.tensor_size = tensor_size
        self.DEVICE = DEVICE
        self.self_attention1 = Self_Attention(tensor_size, num_heads, DEVICE)
        self.self_attention2 = Self_Attention(tensor_size, num_heads, DEVICE)
        self.droput1 = nn.Dropout()
        self.layer_norm1 = nn.LayerNorm(normalized_shape=self.tensor_size)
        self.droput2 = nn.Dropout()
        self.layer_norm2 = nn.LayerNorm(normalized_shape=self.tensor_size)
        self.droput3 = nn.Dropout()
        self.layer_norm3 = nn.LayerNorm(normalized_shape=self.tensor_size)
        self.feed_forward = nn.Sequential(nn.Linear(in_features=self.tensor_size, out_features=feed_forward_size),
                                          nn.ReLU(),
                                          nn.Linear(in_features=feed_forward_size, out_features=self.tensor_size))

    def forward(self, tensor_input, encoder_output):

        attention_output = self.self_attention1(tensor_input, tensor_input, tensor_input)
        attention_output = self.droput1(attention_output)
        output = self.layer_norm1(tensor_input + attention_output)

        attention_output_ = self.self_attention2(output, encoder_output, encoder_output)
        attention_output_ = self.droput2(attention_output_)
        output = self.layer_norm2(attention_output_ + output)

        feed_forward_output = self.feed_forward(output)
        feed_forward_output = self.droput3(feed_forward_output)
        output = self.layer_norm3(output + feed_forward_output)

        return output
```

## Transformer

``` python
class Transformer(nn.Module):

    def __init__(self, tensor_size, num_heads, feed_forward_size, DEVICE, class_num, seq_len, num_layers):
        super().__init__()
        self.tensor_size = tensor_size
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.embedding = nn.Linear(in_features=1, out_features=self.tensor_size)
        self.encoder = [Encoder_layer(tensor_size, num_heads, feed_forward_size, DEVICE).to(DEVICE)
                        for _ in range(num_layers)]
        self.decoder = [Decoder_layer(tensor_size, num_heads, feed_forward_size, DEVICE).to(DEVICE)
                        for _ in range(num_layers)]
        self.linear1 = nn.Linear(in_features=self.tensor_size*seq_len, out_features=128)
        self.linear2 = nn.Linear(in_features=128, out_features=class_num)

    def forward(self, tensor_input):

        output = F.relu(self.embedding(tensor_input.unsqueeze(-1)))

        encoder_output = self.encoder[0](output)
        for i in range(1, self.num_layers):
            encoder_output = self.encoder[i](encoder_output)

        decoder_output = self.decoder[0](output, encoder_output)
        for i in range(1, self.num_layers):
            decoder_output = self.decoder[i](decoder_output, encoder_output)

        output = self.linear1(decoder_output.reshape(-1, self.seq_len*self.tensor_size))
        output = self.linear2(output)

        return output
```

## How to use?

``` python
transformer = Transformer(tensor_size=128, num_heads=2, feed_forward_size=256, DEVICE=DEVICE, class_num=2, seq_len=121,
                          num_layers=5)
```
