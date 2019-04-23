from torch.autograd import Function
a = 1
mingrad = 0.
maxgrad = 1.
         
class Sigmaid(Function):
    @staticmethod
    def forward(ctx, input):
        output = 1 / ( 1 + torch.exp(-a * input))
        #output = input.clamp(min=0)
        ctx.save_for_backward(output)
        return output
      
    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        grad = a * output * (1 - output)
        return torch.clamp(grad, mingrad, maxgrad) * grad_output
class HardSigmoid(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = (1 + input) / 2 
        output = output.clamp(0., 1.)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        output = (input > -1) * (input < 1)
        output = output.float() * 1/2 * grad_output
        return output
if __name__ == "__main__":
    import torch
    #x = torch.Tensor([[-10, 10], [-0.5, 0.5]])
    x = torch.tensor([[-10, 10], [-0.5, 0.5]], requires_grad=True)
    hs = HardSigmoid.apply
    y = torch.sum(hs(x))
    y.backward()
    print(x.grad)