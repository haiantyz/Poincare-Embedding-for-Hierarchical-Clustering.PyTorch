import torch.nn as nn

def build_noise(transitive_closure):
    pass

def noise_sampler(num_samples, x):
    """
     shape:
         x = (batch_size, 1)
    Returns list of int that each u in x is not linked to. shape = (batch_size, num_samples)
    """
    pass
                  
class NEGLoss(nn.Module):
    def __init__(self,
                 num_synsets,
                 transitive_closure):
        super(NEGLoss, self).__init__()
        self.num_synsets = num_synsets
        self.transitive_closure = transitive_closure
        
    def forward(self, inputs, targets):
        
        
    
    
class NCELoss(nn.Module):
    """Noise Contrastive Estimation
    NCE is to eliminate the computational cost of softmax
    normalization.
    Ref:
        X.Chen etal Recurrent neural network language
        model training with noise contrastive estimation
        for speech recognition
        https://core.ac.uk/download/pdf/42338485.pdf
    Attributes:
        emb_size: embedding size
        ntokens: vocabulary size
        noise: the distribution of noise
        noise_ratio: $\frac{#noises}{#real data samples}$ (k in paper)
        norm_term: the normalization term (lnZ in paper)
        size_average: average the loss by batch size
        decoder: the decoder matrix
    Shape:
        - noise: :math:`(V)` where `V = vocabulary size`
        - decoder: :math:`(E, V)` where `E = embedding size`
    """

    def __init__(self,
                 ntokens,
                 emb_size,
                 noise,
                 noise_ratio=10,
                 norm_term=9,
                 size_average=True,
                 decoder_weight=None,
                 ):
        super(NCELoss, self).__init__()

        self.noise = noise
        self.noise_ratio = noise_ratio
        self.norm_term = norm_term
        self.ntokens = ntokens
        self.size_average = size_average
        self.decoder = IndexLinear(emb_size, ntokens)
        # Weight tying
        if decoder_weight:
            self.decoder.weight = decoder_weight

    def forward(self, input, target=None):
        """compute the loss with output and the desired target
        Parameters:
            input: the output of the RNN model, being an predicted embedding
            target: the supervised training label.
        Shape:
            - input: :math:`(N, E)` where `N = number of tokens, E = embedding size`
            - target: :math:`(N)`
        Return:
            the scalar NCELoss Variable ready for backward
        """

        length = target.size(0)
        if self.training:
            assert input.size(0) == target.size(0)

            noise_samples = torch.multinomial(
                self.noise,
                self.noise_ratio,
                replacement=True
            ).unsqueeze(0).repeat(length, 1)
            data_prob, noise_in_data_probs = self._get_prob(input, target.data, noise_samples)
            noise_probs = Variable(
                self.noise[noise_samples.view(-1)].view_as(noise_in_data_probs)
            )

            rnn_loss = torch.log(data_prob / (
                data_prob + self.noise_ratio * Variable(self.noise[target.data]
            )))

            noise_loss = torch.sum(
                torch.log((self.noise_ratio * noise_probs) / (noise_in_data_probs + self.noise_ratio * noise_probs)), 1
            )

            loss = -1 * torch.sum(rnn_loss + noise_loss)

        else:
            out = self.decoder(input, indices=target.unsqueeze(1))
            nll = out.sub(self.norm_term)
            loss = -1 * nll.sum()

        if self.size_average:
            loss = loss / length
        return loss

    def _get_prob(self, embedding, target_idx, noise_idx):
        """Get the NCE estimated probability for target and noise
        Shape:
            - Embedding: :math:`(N, E)`
            - Target_idx: :math:`(N)`
            - Noise_idx: :math:`(N, N_r)` where `N_r = noise ratio`
        """

        embedding = embedding
        indices = Variable(
            torch.cat([target_idx.unsqueeze(1), noise_idx], dim=1)
        )
        probs = self.decoder(embedding, indices)

        probs = probs.sub(self.norm_term).exp()
        return probs[:,0], probs[:,1:]

    
    
    
