"""
    ViT <: TwoHeadNetwork

ViT Network as described in https://arxiv.org/pdf/2010.11929.pdf

Used https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632
as a guide 
"""
mutable struct ViT <: TwoHeadNetwork
  gspec
  hyper
  common
  vhead
  phead
end

@kwdef struct ViTHP
    embed_size :: Int
    num_heads :: Int
    depth :: Int
    patch_size :: Int = 2
end

struct ClsTokenLayer
    cls_tokens
end 

function ClsTokenLayer(embed_size::Int)
    return ClsTokenLayer(randn(1, embed_size) |> gpu)
end

function (m::ClsTokenLayer)(x)
  repeated = repeat(m.cls_tokens, 1, 1, size(x)[end])
  return (CUDA.@allowscalar vcat(repeated, x))
end

Flux.@functor ClsTokenLayer

struct PositionalEmbedding
    embeddings
end 

function PositionalEmbedding(num_pos, embed_size::Int)
    return PositionalEmbedding(randn(num_pos, embed_size) |> gpu)
end

(m::PositionalEmbedding)(x) = m.embeddings .+ x

Flux.@functor PositionalEmbedding

function PatchEmbedding(indim, patch_size, emb_size) 
    flat_embed = function(input)
        h,w,c,b = size(input)
        return reshape(input, h*w, c, b)
    end

    h,w,c = indim
    new_w = (w - patch_size + 2) ÷ 2 + 1
    new_h = (h - patch_size + 2) ÷ 2 + 1
    num_pos = new_h * new_w
    # In connect4 case output is (17,emb_size,b)
    return Chain(
        Conv((patch_size, patch_size), c=>emb_size, stride=patch_size, pad=1),
        flat_embed,
        ClsTokenLayer(emb_size),
        PositionalEmbedding(num_pos+1, emb_size),
        x -> permutedims(x, (2,1,3))
    )
end

#!!!FROM https://github.com/FluxML/Flux.jl/pull/2146/  [modified slightly]

struct MultiHeadAttention{P1, D, P2}
  nheads::Int
  q_proj::P1
  k_proj::P1
  v_proj::P1
  attn_drop::D
  out_proj::P2
end

Flux.@functor MultiHeadAttention

function MultiHeadAttention(dims; 
                      nheads::Int = 8,
                      bias::Bool = false,
                      init = Flux.glorot_uniform,                    
                      dropout_prob = 0.0)

  dims = normalize_mha_dims(dims)
  @assert dims.qk % nheads == 0 "qk_dim should be divisible by nheads"
  @assert dims.v % nheads == 0 "v_dim should be divisible by nheads"
  q_proj = Dense(dims.q_in => dims.qk; bias, init)
  k_proj = Dense(dims.k_in => dims.qk; bias, init)
  v_proj = Dense(dims.v_in => dims.v; bias, init)
  attn_drop = Dropout(dropout_prob)
  out_proj = Dense(dims.v => dims.out; bias, init)
  return MultiHeadAttention(nheads, q_proj, k_proj, v_proj, attn_drop, out_proj)
end

# turns the dims argument into a named tuple
normalize_mha_dims(dims::Int) = 
  (; q_in=dims, k_in=dims, v_in=dims, qk=dims, v=dims, out=dims)

# self-attention
(mha::MultiHeadAttention)(qkv; kws...) = mha(qkv, qkv, qkv; kws...)

# key and value are the same
(mha::MultiHeadAttention)(q, kv; kws...) = mha(q, kv, kv; kws...)

function (mha::MultiHeadAttention)(q_in, k_in, v_in, 
                                  bias=nothing; mask=nothing)
  ## [q_in] = [q_in_dim, q_len, batch_size]
  ## [k_in] = [k_in_dim, kv_len, batch_size] 
  ## [v_in] = [v_in_dim, kv_len, batch_size]
  q = mha.q_proj(q_in)  # [q] = [qk_dim, q_len, batch_size]
  k = mha.k_proj(k_in)  # [k] = [qk_dim, kv_len, batch_size] 
  v = mha.v_proj(v_in)  # [v] = [v_dim, kv_len, batch_size]
  x, _ = dot_product_attention(q, k, v, bias; mha.nheads, mask, fdrop=mha.attn_drop)
  x = mha.out_proj(x)
  # [x] = [out_dim, q_len, batch_size]
  # [α] = [kv_len, q_len, nheads, batch_size]
  return x
end

function EncoderBlock(emb_size, n_heads, mlp_expansion=4)
  mha_out = Chain(
      LayerNorm(emb_size),
      MultiHeadAttention(emb_size, nheads=n_heads)
  )

  mlp_out = Chain(
      LayerNorm(emb_size),
      Dense(emb_size => emb_size*mlp_expansion), 
      gelu,
      Dense(emb_size*mlp_expansion => emb_size)
  )

  return Chain(
      SkipConnection(mha_out, +),
      SkipConnection(mlp_out, +)
  )
end

function Encoder(emb_size, n_heads, depth)
  return Chain([EncoderBlock(emb_size, n_heads)  for i in 1:depth]...)
end

function ViT(gspec::AbstractGameSpec, hyper::ViTHP)
  indim = GI.state_dim(gspec)
  outdim = GI.num_actions(gspec)
  h,w,c = indim
  new_w = (w - hyper.patch_size + 2) ÷ 2 + 1
  new_h = (h - hyper.patch_size + 2) ÷ 2 + 1
  num_pos = (new_h * new_w) + 1
  common = Chain(
    PatchEmbedding(indim, hyper.patch_size, hyper.embed_size),
    Encoder(hyper.embed_size, hyper.num_heads, hyper.depth),
    Flux.flatten
  )

  phead = Chain(
    Dense(num_pos * hyper.embed_size => outdim),
    softmax
  )

  vhead = Chain(
    Dense(num_pos * hyper.embed_size => hyper.embed_size, relu),
    Dense(hyper.embed_size => 1, tanh)
  )
  return ViT(gspec, hyper, common, vhead, phead)
end

function ViT(indim, outdim, hyper::ViTHP)
  h,w,c = indim
  new_w = (w - hyper.patch_size + 2) ÷ 2 + 1
  new_h = (h - hyper.patch_size + 2) ÷ 2 + 1
  num_pos = (new_h * new_w) + 1
  common = Chain(
    PatchEmbedding(indim, hyper.patch_size, hyper.embed_size),
    Encoder(hyper.embed_size, hyper.num_heads, hyper.depth),
    Flux.flatten
  )

  phead = Chain(
    Dense(num_pos * hyper.embed_size => outdim),
    softmax
  )

  vhead = Chain(
    Dense(num_pos * hyper.embed_size => hyper.embed_size, relu),
    Dense(hyper.embed_size => 1, tanh)
  )
  return ViT(nothing, hyper, common, vhead, phead)
end

Network.HyperParams(::Type{ViT}) = ViTHP

function Base.copy(nn::ViT)
  return ViT(
    nn.gspec,
    nn.hyper,
    deepcopy(nn.common),
    deepcopy(nn.vhead),
    deepcopy(nn.phead)
  )
end