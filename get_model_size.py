import utils

model = utils.Net()

param_size = 0
for param in model.parameters():
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

size_all_bits = (param_size + buffer_size) * 8
print(size_all_bits)
