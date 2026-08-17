/* Inference for Llama-3 Transformer model in pure C */

#include "hip/hip_runtime.h"
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <chrono>
#include <memory>
#include <vector>
#include <hip/atomic>
#include <hip/std/memory>
#include <hip/thread>
#ifdef _WIN32
#  include "win.h"
#else
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <unistd.h>
#endif
#include <thrust/copy.h>
#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_ptr.h>
// ----------------------------------------------------------------------------
// Transformer model

#define HIP_CHECK(hip_call)                                                                           \
    do {                                                                                              \
        hipError_t err = hip_call;                                                                    \
        if (err != hipSuccess) {                                                                      \
            fprintf(stderr, "HIP Error: %s at %s:%d\n", hipGetErrorString(err), __FILE__, __LINE__);  \
            exit(EXIT_FAILURE);                                                                       \
        }                                                                                             \
    } while (0)

typedef struct {
  int dim;        // transformer dimension
  int hidden_dim; // for ffn layers
  int n_layers;   // number of layers
  int n_heads;    // number of query heads
  int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
  int vocab_size; // vocabulary size, usually 4096 (byte-level)
  int seq_len;    // max sequence length
} Config;

typedef struct {
  // token embedding table
  float *token_embedding_table; // (vocab_size, dim)
  // weights for rmsnorms
  float *rms_att_weight; // (layer, dim) rmsnorm weights
  float *rms_ffn_weight; // (layer, dim)
  // weights for matmuls. note dim == n_heads * head_size
  float *wq; // (layer, dim, n_heads * head_size)
  float *wk; // (layer, dim, n_kv_heads * head_size)
  float *wv; // (layer, dim, n_kv_heads * head_size)
  float *wo; // (layer, n_heads * head_size, dim)
  // weights for ffn
  float *w1; // (layer, hidden_dim, dim)
  float *w2; // (layer, dim, hidden_dim)
  float *w3; // (layer, hidden_dim, dim)
  // final rmsnorm
  float *rms_final_weight; // (dim,)
  // (optional) classifier weights for the logits, on the last layer
  float *wcls;
} TransformerWeights;

typedef struct {
  // current wave of activations
  hip::std::unique_ptr<float[]> x;      // activation at current time stamp (dim,)
  hip::std::unique_ptr<float[]> xb;     // same, but inside a residual branch (dim,)
  hip::std::unique_ptr<float[]> xb2;    // an additional buffer just for convenience (dim,)
  hip::std::unique_ptr<float[]> hb;     // buffer for hidden dimension in the ffn (hidden_dim,)
  hip::std::unique_ptr<float[]> hb2;    // buffer for hidden dimension in the ffn (hidden_dim,)
  hip::std::unique_ptr<float[]> q;      // query (dim,)
  float *k;                        // key (dim,)
  float *v;                        // value (dim,)
  hip::std::unique_ptr<float[]> att;    // buffer for scores/attention values (n_heads, seq_len)
  hip::std::unique_ptr<float[]> logits; // output logits
  // kv cache
  hip::std::unique_ptr<float[]> key_cache;   // (layer, seq_len, dim)
  hip::std::unique_ptr<float[]> value_cache; // (layer, seq_len, dim)
  float *logits_raw;
} RunState;

// Device-resident Transformer structure
typedef struct {
  Config config;              // the hyperparameters of the architecture (the blueprint)
  TransformerWeights weights; // the weights of the model
  RunState state;             // buffers for the "wave" of activations in the forward pass
} Transformer;

// HIP stream for async operations
hipStream_t g_stream;

RunState __device__ malloc_run_state(Config *p) {
  RunState s;
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  s.x = hip::std::make_unique<float[]>(p->dim);
  s.xb = hip::std::make_unique<float[]>(p->dim);
  s.xb2 = hip::std::make_unique<float[]>(p->dim);
  s.hb = hip::std::make_unique<float[]>(p->hidden_dim);
  s.hb2 = hip::std::make_unique<float[]>(p->hidden_dim);
  s.q = hip::std::make_unique<float[]>(p->dim);
  s.key_cache = hip::std::make_unique<float[]>(p->n_layers * p->seq_len * kv_dim);
  s.value_cache = hip::std::make_unique<float[]>(p->n_layers * p->seq_len * kv_dim);
  s.att = hip::std::make_unique<float[]>(p->n_heads * p->seq_len);
  s.logits = hip::std::make_unique<float[]>(p->vocab_size);
  s.logits_raw = s.logits.get();
  // ensure all mallocs went fine
  if (!s.x || !s.xb || !s.xb2 || !s.hb || !s.hb2 || !s.q || !s.key_cache || !s.value_cache || !s.att || !s.logits) {
    printf("Memory allocation failed!\n");
    return {};
  }
  return s;
}

TransformerWeights __device__ memory_map_weights(Config *p, float *ptr, int shared_weights) {
  TransformerWeights w;
  int head_size = p->dim / p->n_heads;
  // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
  unsigned long long n_layers = p->n_layers;
  w.token_embedding_table = ptr;
  ptr += p->vocab_size * p->dim;
  w.rms_att_weight = ptr;
  ptr += n_layers * p->dim;
  w.wq = ptr;
  ptr += n_layers * p->dim * (p->n_heads * head_size);
  w.wk = ptr;
  ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
  w.wv = ptr;
  ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
  w.wo = ptr;
  ptr += n_layers * (p->n_heads * head_size) * p->dim;
  w.rms_ffn_weight = ptr;
  ptr += n_layers * p->dim;
  w.w1 = ptr;
  ptr += n_layers * p->dim * p->hidden_dim;
  w.w2 = ptr;
  ptr += n_layers * p->hidden_dim * p->dim;
  w.w3 = ptr;
  ptr += n_layers * p->dim * p->hidden_dim;
  w.rms_final_weight = ptr;
  ptr += p->dim;
  ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
  ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
  w.wcls = shared_weights ? w.token_embedding_table : ptr;
  return w;
}

struct Checkpoint {
  Config config;
  float *weights_ptr;
  int shared_weights;
};

Checkpoint read_checkpoint(const char *checkpoint_path) {
  FILE *file = fopen(checkpoint_path, "rb");
  if (!file) {
    fprintf(stderr, "Couldn't open file %s\n", checkpoint_path);
    exit(EXIT_FAILURE);
  }

  Config config;
  // read in the config header
  if (fread(&config, sizeof(Config), 1, file) != 1) {
    exit(EXIT_FAILURE);
  }

  // negative vocab size is hacky way of signaling unshared weights. bit yikes.
  int shared_weights = config.vocab_size > 0 ? 1 : 0;
  config.vocab_size = abs(config.vocab_size);

  // figure out the file size
  fseek(file, 0, SEEK_END); // move file pointer to end of file
  long long file_size = ftell(file); // win.h redefines ftell to _ftelli64 on Windows; long is 32-bit on Windows so must use long long for >2GB files
  fclose(file);

  // memory map the Transformer weights into the data pointer
  int fd = open(checkpoint_path, O_RDONLY); // open in read only mode
  if (fd == -1) {
    fprintf(stderr, "open failed!\n");
    exit(EXIT_FAILURE);
  }

  float *data = reinterpret_cast<float*>(mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0));
  if (data == MAP_FAILED) {
    fprintf(stderr, "mmap failed!\n");
    exit(EXIT_FAILURE);
  }

  // Copy weights to device
  size_t weights_size = file_size - sizeof(Config);
  size_t num_floats = weights_size / sizeof(float);
  thrust::device_ptr<float> weights_ptr_dev = thrust::device_malloc<float>(num_floats);
  float *weights_ptr = thrust::raw_pointer_cast(weights_ptr_dev);
  thrust::copy_n(reinterpret_cast<float*>(((char*)data) + sizeof(Config)), num_floats, weights_ptr_dev);

  // Clean up mmap immediately - we don't need it anymore
  munmap(data, file_size);
  close(fd);

  return {config, weights_ptr, shared_weights};
}

std::tuple<Transformer*, Config, float*> build_transformer(const char *checkpoint_path) {
  // read in the Config and the Weights from the checkpoint
  auto [config, weights_ptr, shared_weights] = read_checkpoint(checkpoint_path);

  // Allocate device-resident Transformer
  thrust::device_ptr<Transformer> transformer_ptr_dev = thrust::device_malloc<Transformer>(1);
  Transformer *transformer = thrust::raw_pointer_cast(transformer_ptr_dev);

  // Copy config to device
  thrust::device_ptr<Config> config_ptr_dev(&transformer->config);
  thrust::copy_n(&config, 1, config_ptr_dev);

  // Initialize weights and state on device
  hip::wthread(
    [] __device__ (Transformer *t, float *weights_ptr, int shared_weights) {
      // Set up weight pointers within the device memory buffer
      t->weights = memory_map_weights(&t->config, weights_ptr, shared_weights);

      // Initialize RunState
      // Use placement-new so we don't try to assign to unique_ptrs that think they already point at something
      new (&t->state) RunState(malloc_run_state(&t->config));
    }, transformer, weights_ptr, shared_weights
  ).join();

  return {transformer, config, weights_ptr};
}

void free_transformer(Transformer *transformer, float *weights_ptr) {
  // Destruct RunState on device to free unique_ptr allocations
  hip::wthread([] __device__ (Transformer *t) {
      t->state.~RunState();
    }, transformer
  ).join();

  // Free weights buffer
  thrust::device_free(thrust::device_pointer_cast(weights_ptr));

  // Free transformer struct
  thrust::device_free(thrust::device_pointer_cast(transformer));
}

// ----------------------------------------------------------------------------
// GPU-specific helper functions for parallel execution

constexpr int __host__ __device__ get_chunk_size(int arr_size, int tid, int num_threads) {
  return (tid < arr_size % num_threads) ? (arr_size / num_threads + 1) : (arr_size / num_threads);
}

constexpr int __host__ __device__ get_start_row(int arr_size, int tid, int num_threads) {
  int chunk_size = get_chunk_size(arr_size, tid, num_threads);
  return (tid < arr_size % num_threads) ? (tid * chunk_size) : (tid * chunk_size + arr_size % num_threads);
}

// Spin-lock barrier for n threads. barrier must be big enough to count 3*nThreads - 1 without overflowing.
template <uint32_t nThreads>
uint32_t __device__ barrier_wait()
{
  static __device__ hip::atomic<uint32_t, hip::thread_scope_device> barrier = 0;
  __shared__ uint32_t v;
  if (threadIdx.x == 0) {
    v = ++barrier;
    assert(v < 3 * nThreads);
  }
  __threadfence();
  __syncthreads();
  if (v == nThreads)
    return 0;
  if (v == 2 * nThreads) {
    if (threadIdx.x == 0) {
      barrier -= 2 * nThreads;
    }
    __threadfence();
    __syncthreads();
    return 0;
  }
  while ((barrier/nThreads & 1) == (v/nThreads & 1))
  {}
  __threadfence();
  __syncthreads();
  return v % nThreads;
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

void __device__ rmsnorm(float *o, float *x, float *weight, int size) {
  // calculate sum of squares
  float ss = 0.0f;
  for (int j = 0; j < size; j++) {
    ss += x[j] * x[j];
  }
  ss /= size;
  ss += 1e-5f;
  ss = 1.0f / sqrtf(ss);
  // normalize and scale
  for (int j = 0; j < size; j++) {
    o[j] = weight[j] * (ss * x[j]);
  }
}

void __host__ __device__ softmax(float *x, int size) {
  // find max value (for numerical stability)
  float max_val = x[0];
  for (int i = 1; i < size; i++) {
    if (x[i] > max_val) {
      max_val = x[i];
    }
  }
  // exp and sum
  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  // normalize
  for (int i = 0; i < size; i++) {
    x[i] /= sum;
  }
}

void __device__ matmul(float *xout, float *x, float *w, int n, int d, int tid, int num_threads) {
  // W (d,n) @ x (n,) -> xout (d,)
  // by far the most amount of time is spent inside this little function
  if (!d)
    return;

  const int chunk_size = get_chunk_size(d, tid, num_threads);
  const int start_row = get_start_row(d, tid, num_threads);

  for (int i = start_row; i < start_row + chunk_size; ++i) {
    float val = 0.0f;
    for (int j = 0; j < n; ++j){
      val += w[i * n + j] * x[j];
    }
    xout[i] = val;
  }
}

std::unique_ptr<float[]> forward(Transformer *transformer, int token, int pos, int vocab_size) {
  constexpr size_t num_threads = 128;
  std::vector<hip::wthread> threads{num_threads};

  for (int tid = 0; tid < num_threads; ++tid) {
    threads[tid] = hip::wthread([] __device__ (Transformer *t, int token, int pos, int tid) {
      Config *p = &t->config;
      TransformerWeights *w = &t->weights;
      RunState *s = &t->state;

      float *const x = s->x.get();
      const int dim = p->dim;
      const int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
      const int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
      const int hidden_dim = p->hidden_dim;
      const int head_size = dim / p->n_heads;

      // copy the token embedding into x
      if (tid == 0) {
        float *content_row = w->token_embedding_table + token * dim;
        memcpy(x, content_row, dim * sizeof(*x));
      }

      // forward all the layers
      for (unsigned long long l = 0; l < p->n_layers; l++) {
        const int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        // Regardless of whether we are entering the loop for the first time or we just finished
        // looping, all the work since the last barrier_wait has been within an `if (tid == 0)`.
        // Thus, no barrier_wait is needed here.

        // attention rmsnorm
        if (tid == 0) {
          rmsnorm(s->xb.get(), x, w->rms_att_weight + l * dim, dim);
          // key and value point to the kv cache
          s->k = s->key_cache.get() + loff + pos * kv_dim;
          s->v = s->value_cache.get() + loff + pos * kv_dim;
        }
        barrier_wait<num_threads>();

        // qkv matmuls for this position (all threads participate)
        matmul(s->q.get(), s->xb.get(), w->wq + l * dim * dim, dim, dim, tid, num_threads);
        matmul(s->k, s->xb.get(), w->wk + l * dim * kv_dim, dim, kv_dim, tid, num_threads);
        matmul(s->v, s->xb.get(), w->wv + l * dim * kv_dim, dim, kv_dim, tid, num_threads);
        barrier_wait<num_threads>();

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for (int i = tid; i < p->n_heads; i += num_threads) {
          for (int j = 0; j < head_size; j += 2) {
            float freq = 1.0f / powf(500000.0f, (float)j / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            float q0 = s->q[i * head_size + j];
            float q1 = s->q[i * head_size + j + 1];
            s->q[i * head_size + j] = q0 * fcr - q1 * fci;
            s->q[i * head_size + j + 1] = q0 * fci + q1 * fcr;
            if (i < p->n_kv_heads) {
              float k0 = s->k[i * head_size + j];
              float k1 = s->k[i * head_size + j + 1];
              s->k[i * head_size + j] = k0 * fcr - k1 * fci;
              s->k[i * head_size + j + 1] = k0 * fci + k1 * fcr;
            }
          }
        }
        barrier_wait<num_threads>();

        // multihead attention. iterate over all heads
        for (int h = tid; h < p->n_heads; h += num_threads) {
          // get the query vector for this head
          float *q = s->q.get() + h * head_size;
          // attention scores for this head
          float *att = s->att.get() + h * p->seq_len;
          // iterate over all timesteps, including the current one
          for (int t = 0; t <= pos; t++) {
            // get the key vector for this head and at this timestep
            float *k = s->key_cache.get() + loff + t * kv_dim + (h / kv_mul) * head_size;
            // calculate the attention score as the dot product of q and k
            float score = 0.0f;
            for (int i = 0; i < head_size; i++) {
              score += q[i] * k[i];
            }
            score /= sqrtf(head_size);
            // save the score to the attention buffer
            att[t] = score;
          }

          // softmax the scores to get attention weights, from 0..pos inclusively
          softmax(att, pos + 1);

          // weighted sum of the values, store back into xb
          float *xb = s->xb.get() + h * head_size;
          for (int i = 0; i < head_size; i++) {
            float val = 0.0f;
            for (int t = 0; t <= pos; t++) {
              // get the value vector for this head and at this timestep
              float *v = s->value_cache.get() + loff + t * kv_dim + (h / kv_mul) * head_size;
              // get the attention weight for this timestep
              float a = att[t];
              // accumulate the weighted value
              val += a * v[i];
            }
            xb[i] = val;
          }
        }
        barrier_wait<num_threads>();

        // final matmul to get the output of the attention
        matmul(s->xb2.get(), s->xb.get(), w->wo + l * dim * dim, dim, dim, tid, num_threads);

        barrier_wait<num_threads>();
        // residual connection back into x
        if (tid == 0) {
          for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
          }
          // ffn rmsnorm
          rmsnorm(s->xb.get(), x, w->rms_ffn_weight + l * dim, dim);
        }
        barrier_wait<num_threads>();

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(s->hb.get(), s->xb.get(), w->w1 + l * dim * hidden_dim, dim, hidden_dim, tid, num_threads);
        matmul(s->hb2.get(), s->xb.get(), w->w3 + l * dim * hidden_dim, dim, hidden_dim, tid, num_threads);

        // SwiGLU chunks
        const int chunk_size = get_chunk_size(hidden_dim, tid, num_threads);
        const int start_row = get_start_row(hidden_dim, tid, num_threads);

        // No barrier_wait is needed because the preceding two matmul calls use the same chunking
        // as we will for SwiGLU, thus each thread is only reading from array entries it wrote to.
        // SwiGLU non-linearity
        for (int i = start_row; i < start_row + chunk_size; i++) {
          float val = s->hb[i];
          // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
          val *= (1.0f / (1.0f + expf(-val)));
          // elementwise multiply with w3(x)
          val *= s->hb2[i];
          s->hb[i] = val;
        }
        barrier_wait<num_threads>();

        // final matmul to get the output of the ffn
        matmul(s->xb.get(), s->hb.get(), w->w2 + l * dim * hidden_dim, hidden_dim, dim, tid, num_threads);

        barrier_wait<num_threads>();
        // residual connection
        if (tid == 0) {
          for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
          }
        }
      }

      // final rmsnorm
      // All the work since the last barrier_wait (in the for loop) was within an `if (tid == 0)`.
      // Thus, no barrier_wait is needed and a simple __syncthreads() will be sufficient.
      if (tid == 0) {
        __syncthreads();
        rmsnorm(x, x, w->rms_final_weight, dim);
      }
      barrier_wait<num_threads>();

      // classifier into logits
      matmul(s->logits.get(), x, w->wcls, p->dim, p->vocab_size, tid, num_threads);
    }, transformer, token, pos, tid);
  }

  // Allocate host logits buffer
  std::unique_ptr<float[]> logits = std::make_unique<float[]>(vocab_size);

  for (auto &t : threads) {
    t.join();
  }

  // Copy logits back to host. We can't use thrust APIs here because there is still a hip::wthread on
  // the stack from generate/chat.
  float *logits_device_ptr;
  HIP_CHECK(hipMemcpyAsync(&logits_device_ptr, &transformer->state.logits_raw, sizeof(float*), hipMemcpyDeviceToHost, g_stream));
  HIP_CHECK(hipStreamSynchronize(g_stream));
  HIP_CHECK(hipMemcpyAsync(logits.get(), logits_device_ptr, vocab_size * sizeof(float), hipMemcpyDeviceToHost, g_stream));
  HIP_CHECK(hipStreamSynchronize(g_stream));
  return logits;
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
  char *str;
  int id;
} TokenIndex;

typedef struct {
  char **vocab;
  float *vocab_scores;
  TokenIndex *sorted_vocab;
  int vocab_size;
  unsigned int max_token_length;
  unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

int compare_tokens(const void *a, const void *b) { return strcmp(((TokenIndex *)a)->str, ((TokenIndex *)b)->str); }

void build_tokenizer(Tokenizer *t, const char *tokenizer_path, int vocab_size) {
  // i should have written the vocab_size into the tokenizer file... sigh
  t->vocab_size = vocab_size;
  // malloc space to hold the scores and the strings
  t->vocab = (char **)malloc(vocab_size * sizeof(char *));
  t->vocab_scores = (float *)malloc(vocab_size * sizeof(float));
  t->sorted_vocab = NULL; // initialized lazily
  for (int i = 0; i < 256; i++) {
    t->byte_pieces[i * 2] = (unsigned char)i;
    t->byte_pieces[i * 2 + 1] = '\0';
  }
  // read in the file
  FILE *file = fopen(tokenizer_path, "rb");
  if (!file) {
    fprintf(stderr, "couldn't load %s\n", tokenizer_path);
    exit(EXIT_FAILURE);
  }
  if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) {
    fprintf(stderr, "failed read\n");
    exit(EXIT_FAILURE);
  }
  int len;
  for (int i = 0; i < vocab_size; i++) {
    if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) {
      fprintf(stderr, "failed read\n");
      exit(EXIT_FAILURE);
    }
    if (fread(&len, sizeof(int), 1, file) != 1) {
      fprintf(stderr, "failed read\n");
      exit(EXIT_FAILURE);
    }
    t->vocab[i] = (char *)malloc(len + 1);
    if (fread(t->vocab[i], len, 1, file) != 1) {
      fprintf(stderr, "failed read\n");
      exit(EXIT_FAILURE);
    }
    t->vocab[i][len] = '\0'; // add the string terminating token
  }
  fclose(file);
}

void free_tokenizer(Tokenizer *t) {
  for (int i = 0; i < t->vocab_size; i++) {
    free(t->vocab[i]);
  }
  free(t->vocab);
  free(t->vocab_scores);
  free(t->sorted_vocab);
}

char *decode(Tokenizer *t, int prev_token, int token) {
  char *piece = t->vocab[token];

  // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
  // parse this and convert and return the actual byte
  unsigned char byte_val;
  if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
    piece = (char *)t->byte_pieces + byte_val * 2;
  }
  return piece;
}

void safe_printf(char *piece) {
  // piece might be a raw byte token, and we only want to print printable chars or whitespace
  // because some of the other bytes can be various control codes, backspace, etc.
  if (piece == NULL) {
    return;
  }
  if (piece[0] == '\0') {
    return;
  }
  if (piece[1] == '\0') {
    unsigned char byte_val = piece[0];
    if (!(isprint(byte_val) || isspace(byte_val))) {
      return; // bad byte, don't print it
    }
  }
  printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
  // efficiently find the perfect match for str in vocab, return its index or -1 if not found
  TokenIndex tok = {.str = str}; // acts as the key to search for
  TokenIndex *res = reinterpret_cast<TokenIndex*>(bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens));
  return res != NULL ? res->id : -1;
}

void encode(Tokenizer *t, const char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
  // encode the string text (input) into an upper-bound preallocated tokens[] array
  // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
  if (text == NULL) {
    fprintf(stderr, "cannot encode NULL text\n");
    exit(EXIT_FAILURE);
  }

  if (t->sorted_vocab == NULL) {
    // lazily malloc and sort the vocabulary
    t->sorted_vocab = reinterpret_cast<TokenIndex*>(malloc(t->vocab_size * sizeof(TokenIndex)));
    for (int i = 0; i < t->vocab_size; i++) {
      t->sorted_vocab[i].str = t->vocab[i];
      t->sorted_vocab[i].id = i;
    }
    qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
  }

  // create a temporary buffer that will store merge candidates of always two consecutive tokens
  // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
  char *str_buffer = reinterpret_cast<char*>(malloc((t->max_token_length * 2 + 1 + 2) * sizeof(char)));
  size_t str_len = 0;

  // start at 0 tokens
  *n_tokens = 0;

  // add optional BOS (=128000) token, if desired
  if (bos)
    tokens[(*n_tokens)++] = 128000;

  // add_dummy_prefix is true by default
  // so prepend a dummy prefix token to the input string, but only if text != ""
  // TODO: pretty sure this isn't correct in the general case but I don't have the
  // energy to read more of the sentencepiece code to figure out what it's doing

  // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
  // Code point ↔ UTF-8 conversion
  // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
  // U+0000	U+007F	    0xxxxxxx
  // U+0080	U+07FF	    110xxxxx	10xxxxxx
  // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
  // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

  // process the raw (UTF-8) byte sequence of the input string
  for (const char *c = text; *c != '\0'; c++) {

    // reset buffer if the current byte is ASCII or a leading byte
    // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
    // 0x80 is 10000000
    // in UTF-8, all continuation bytes start with "10" in first two bits
    // so in English this is: "if this byte is not a continuation byte"
    if ((*c & 0xC0) != 0x80) {
      // this byte must be either a leading byte (11...) or an ASCII char (0x...)
      // => reset our location, as we're starting a new UTF-8 codepoint
      str_len = 0;
    }

    // append the current byte to the buffer
    str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
    str_buffer[str_len] = '\0';

    // while the next character is a continuation byte, continue appending
    // but if there are too many of them, just stop to avoid overruning str_buffer size.
    if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) {
      continue;
    }

    // ok c+1 is not a continuation byte, so we've read in a full codepoint
    int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

    if (id != -1) {
      // we found this codepoint in vocab, add it as a token
      tokens[(*n_tokens)++] = id;
    } else {
      // byte_fallback encoding: just encode each byte as a token
      // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
      // so the individual bytes only start at index 3
      for (size_t i = 0; i < str_len; i++) {
        tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
      }
    }
    str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
  }

  // merge the best consecutive pair or triple each iteration, according to the scores in vocab_scores
  while (1) {
    float best_score = -1e10;
    int best_id = -1;
    int best_idx = -1;
    int best_len = 2; // length of the best merge sequence (2 for pair, 3 for triple)

    // first, try to find the best pair to merge
    for (int i = 0; i < (*n_tokens - 1); i++) {
      // check if we can merge the pair (tokens[i], tokens[i+1])
      sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
      int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
      if (id != -1 && t->vocab_scores[id] > best_score) {
        // this merge pair exists in vocab! record its score and position
        best_score = t->vocab_scores[id];
        best_id = id;
        best_idx = i;
      }
    }

    // if no pair was found, try to find the best triple to merge
    if (best_idx == -1) {
      for (int i = 0; i < (*n_tokens - 2); i++) {
        // check if we can merge the triple (tokens[i], tokens[i+1], tokens[i+2])
        sprintf(str_buffer, "%s%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]], t->vocab[tokens[i + 2]]);
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
        if (id != -1 && t->vocab_scores[id] > best_score) {
          // this merge triple exists in vocab! record its score and position
          best_score = t->vocab_scores[id];
          best_id = id;
          best_idx = i;
          best_len = 3;
        }
      }
    }

    if (best_idx == -1) {
      break; // we couldn't find any more pairs or triples to merge, so we're done
    }

    // merge the consecutive pair or triple (best_idx, best_idx+1[, best_idx+2]) into new token best_id
    tokens[best_idx] = best_id;
    // delete token(s) at position best_idx+1 (and optionally best_idx+2), shift the entire sequence back
    for (int i = best_idx + 1; i < (*n_tokens - best_len + 1); i++) {
      tokens[i] = tokens[i + best_len - 1];
    }
    (*n_tokens) -= (best_len - 1); // token length decreased by the number of merged tokens minus one
  }

  // add optional EOS (=128001) token, if desired
  if (eos)
    tokens[(*n_tokens)++] = 128001;

  free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
  float prob;
  int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
  int vocab_size;
  ProbIndex *probindex; // buffer used in top-p sampling
  float temperature;
  float topp;
  unsigned long long rng_state;
} Sampler;

int sample_argmax(float *probabilities, int n) {
  // return the index that has the highest probability
  int max_i = 0;
  float max_p = probabilities[0];
  for (int i = 1; i < n; i++) {
    if (probabilities[i] > max_p) {
      max_i = i;
      max_p = probabilities[i];
    }
  }
  return max_i;
}

int sample_mult(float *probabilities, int n, float coin) {
  // sample index from probabilities (they must sum to 1!)
  // coin is a random number in [0, 1), usually from random_f32()
  float cdf = 0.0f;
  for (int i = 0; i < n; i++) {
    cdf += probabilities[i];
    if (coin < cdf) {
      return i;
    }
  }
  return n - 1; // in case of rounding errors
}

int compare(const void *a, const void *b) {
  ProbIndex *a_ = (ProbIndex *)a;
  ProbIndex *b_ = (ProbIndex *)b;
  if (a_->prob > b_->prob)
    return -1;
  if (a_->prob < b_->prob)
    return 1;
  return 0;
}

int sample_topp(float *probabilities, int n, float topp, ProbIndex *probindex, float coin) {
  // top-p sampling (or "nucleus sampling") samples from the smallest set of
  // tokens that exceed probability topp. This way we never sample tokens that
  // have very low probabilities and are less likely to go "off the rails".
  // coin is a random number in [0, 1), usually from random_f32()

  int n0 = 0;
  // quicksort indices in descending order of probabilities
  // values smaller than (1 - topp) / (n - 1) cannot be part of the result
  // so for efficiency we crop these out as candidates before sorting
  const float cutoff = (1.0f - topp) / (n - 1);
  for (int i = 0; i < n; i++) {
    if (probabilities[i] >= cutoff) {
      probindex[n0].index = i;
      probindex[n0].prob = probabilities[i];
      n0++;
    }
  }
  qsort(probindex, n0, sizeof(ProbIndex), compare);

  // truncate the list where cumulative probability exceeds topp
  float cumulative_prob = 0.0f;
  int last_idx = n0 - 1; // in case of rounding errors consider all elements
  for (int i = 0; i < n0; i++) {
    cumulative_prob += probindex[i].prob;
    if (cumulative_prob > topp) {
      last_idx = i;
      break; // we've exceeded topp by including last_idx
    }
  }

  // sample from the truncated list
  float r = coin * cumulative_prob;
  float cdf = 0.0f;
  for (int i = 0; i <= last_idx; i++) {
    cdf += probindex[i].prob;
    if (r < cdf) {
      return probindex[i].index;
    }
  }
  return probindex[last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler *sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
  sampler->vocab_size = vocab_size;
  sampler->temperature = temperature;
  sampler->topp = topp;
  sampler->rng_state = rng_seed;
  // buffer only used with nucleus sampling; may not need but it's ~small
  sampler->probindex = reinterpret_cast<ProbIndex*>(malloc(sampler->vocab_size * sizeof(ProbIndex)));
}

void free_sampler(Sampler *sampler) { free(sampler->probindex); }

unsigned int random_u32(unsigned long long *state) {
  // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
  *state ^= *state >> 12;
  *state ^= *state << 25;
  *state ^= *state >> 27;
  return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
  return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler *sampler, float *logits) {
  // sample the token given the logits and some hyperparameters
  int next;
  if (sampler->temperature == 0.0f) {
    // greedy argmax sampling: take the token with the highest probability
    next = sample_argmax(logits, sampler->vocab_size);
  } else {
    // apply the temperature to the logits
    for (int q = 0; q < sampler->vocab_size; q++) {
      logits[q] /= sampler->temperature;
    }
    // apply softmax to the logits to get the probabilities for next token
    softmax(logits, sampler->vocab_size);
    // flip a (float) coin (this is our source of entropy for sampling)
    float coin = random_f32(&sampler->rng_state);
    // we sample from this distribution to get the next token
    if (sampler->topp <= 0 || sampler->topp >= 1) {
      // simply sample from the predicted probability distribution
      next = sample_mult(logits, sampler->vocab_size, coin);
    } else {
      // top-p (nucleus) sampling, clamping the least likely tokens to zero
      next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
    }
  }
  return next;
}

// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms() {
  // return time in milliseconds, for benchmarking the model speed
  return static_cast<long>(std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count());
}

// ----------------------------------------------------------------------------
// generation loop

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, const char *prompt, int steps) {
  const char *empty_prompt = "";
  if (prompt == NULL) {
    prompt = empty_prompt;
  }

  // encode the (string) prompt into tokens sequence
  int num_prompt_tokens = 0;
  int *prompt_tokens = (int *)malloc((strlen(prompt) + 3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
  encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
  if (num_prompt_tokens < 1) {
    fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
    exit(EXIT_FAILURE);
  }

  // start the main loop
  long start = 0;               // used to time our code, only initialized after first iteration
  int next;                     // will store the next token in the sequence
  int token = prompt_tokens[0]; // kick off with the first token in the prompt
  int pos = 0;                  // position in the sequence

  // Constructing a dummy hip::wthread keeps the persistent scheduler kernel alive across iterations
  // of the while loop. This prevents us from repeatedly spinning up/down every time we call forward
  hip::wthread _;

  while (pos < steps) {

    // forward the transformer to get logits for the next token
    std::unique_ptr<float[]> logits = forward(transformer, token, pos, sampler->vocab_size);

    // advance the state machine
    if (pos < num_prompt_tokens - 1) {
      // if we are still processing the input prompt, force the next prompt token
      next = prompt_tokens[pos + 1];
    } else {
      // otherwise sample the next token from the logits
      next = sample(sampler, logits.get());
    }
    pos++;

    // data-dependent terminating condition: the BOS (=1) token delimits sequences
    if ((next == 128001 || next == 128009) && pos > num_prompt_tokens)
      break;
    // print the token as string, decode it with the Tokenizer object
    char *piece = decode(tokenizer, token, next);
    safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
    fflush(stdout);
    token = next;

    // init the timer here because the first iteration can be slower
    if (start == 0) {
      start = time_in_ms();
    }
  }
  printf("\n");

  // report achieved tok/s (pos-1 because the timer starts after first iteration)
  if (pos > 1) {
    long end = time_in_ms();
    fprintf(stderr, "achieved tok/s: %f\n", (pos - 1) / (double)(end - start) * 1000);
  }

  free(prompt_tokens);
}

void read_stdin(const char *guide, char *buffer, size_t bufsize) {
  // read a line from stdin, up to but not including \n
  printf("%s", guide);
  if (fgets(buffer, bufsize, stdin) != NULL) {
    size_t len = strlen(buffer);
    if (len > 0 && buffer[len - 1] == '\n') {
      buffer[len - 1] = '\0'; // strip newline
    }
  }
}

// ----------------------------------------------------------------------------
// chat loop
// I manually inspected the tokens for a few chat conversations compared to
// python reference and that seemed ok, but this was not thoroughly tested and
// is not safely implemented, it's more a proof of concept atm.

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *cli_user_prompt, char *cli_system_prompt, int steps) {

  // buffers for reading the system prompt and user prompt from stdin
  // you'll notice they are somewhat haphazardly and unsafely set atm
  char *system_prompt = (char *)malloc(32768 * sizeof(char));
  char *user_prompt = (char *)malloc(32768 * sizeof(char));
  int num_prompt_tokens = 0;
  int *prompt_tokens = (int *)malloc(32768 * sizeof(int));
  int *system_prompt_tokens = (int *)malloc(32768 * sizeof(int));
  int *user_prompt_tokens = (int *)malloc(32768 * sizeof(int));
  int user_idx = 0;

  // start the main loop
  int8_t user_turn = 1; // user starts
  int next;             // will store the next token in the sequence
  int token;            // stores the current token to feed into the transformer

  int pos = 0; // position in the sequence

  // Constructing a dummy hip::wthread keeps the persistent scheduler kernel alive across iterations
  // of the while loop. This prevents us from repeatedly spinning up/down every time we call forward
  hip::wthread _;

  while (pos < steps) {

    // when it is the user's turn to contribute tokens to the dialog...
    if (user_turn) {
      // get the (optional) system prompt at position 0
      if (pos == 0) {
        // at position 0, the user can also contribute a system prompt
        prompt_tokens[num_prompt_tokens++] = 128000; // "<|begin_of_text|>"
        prompt_tokens[num_prompt_tokens++] = 128006; // "<|start_header_id|>"
        prompt_tokens[num_prompt_tokens++] = 9125;   // "system"
        prompt_tokens[num_prompt_tokens++] = 128007; // "<|end_header_id|>"
        prompt_tokens[num_prompt_tokens++] = 271;    // "\n\n"
        if (cli_system_prompt == NULL) {
          // system prompt was not passed in, attempt to get it from stdin
          read_stdin("Enter system prompt (optional): ", system_prompt, 32768);
        } else {
          // system prompt was passed in, use it
          strcpy(system_prompt, cli_system_prompt);
        }
        if (system_prompt != NULL) {
          int num_system_prompt_tokens = 0;
          encode(tokenizer, system_prompt, 0, 0, system_prompt_tokens, &num_system_prompt_tokens);
          for (int i = 0; i < num_system_prompt_tokens; i++) {
            prompt_tokens[num_prompt_tokens++] = system_prompt_tokens[i];
          }
        }
        prompt_tokens[num_prompt_tokens++] = 128009; // "<|eot_id|>"
      } else {
        num_prompt_tokens = 0;
      }
      prompt_tokens[num_prompt_tokens++] = 128006; // "<|start_header_id|>"
      prompt_tokens[num_prompt_tokens++] = 882;    // "user"
      prompt_tokens[num_prompt_tokens++] = 128007; // "<|end_header_id|>"
      prompt_tokens[num_prompt_tokens++] = 271;    // "\n\n"
      // get the user prompt
      if (pos == 0 && cli_user_prompt != NULL) {
        // user prompt for position 0 was passed in, use it
        strcpy(user_prompt, cli_user_prompt);
      } else {
        // otherwise get user prompt from stdin
        read_stdin("User (or exit): ", user_prompt, 32768);
        if (strcmp(user_prompt, "exit") == 0)
          break;
      }
      int num_user_prompt_tokens = 0;
      // encode the user prompt into tokens
      encode(tokenizer, user_prompt, 0, 0, user_prompt_tokens, &num_user_prompt_tokens);
      for (int i = 0; i < num_user_prompt_tokens; i++) {
        prompt_tokens[num_prompt_tokens++] = user_prompt_tokens[i];
      }
      prompt_tokens[num_prompt_tokens++] = 128009; // "<|eot_id|>"
      prompt_tokens[num_prompt_tokens++] = 128006; // "<|start_header_id|>"
      prompt_tokens[num_prompt_tokens++] = 78191;  // "assistant"
      prompt_tokens[num_prompt_tokens++] = 128007; // "<|end_header_id|>"
      prompt_tokens[num_prompt_tokens++] = 271;    // "\n\n"

      user_idx = 0; // reset the user index
      user_turn = 0;
      printf("Assistant: ");
    }

    // determine the token to pass into the transformer next
    if (user_idx < num_prompt_tokens) {
      // if we are still processing the input prompt, force the next prompt token
      token = prompt_tokens[user_idx++];
    } else {
      // otherwise use the next token sampled from previous turn
      token = next;
    }
    // EOS (=128009) token ends the Assistant turn
    if (user_idx >= num_prompt_tokens && (token == 128009 || token == 128001)) {
      user_turn = 1;
    }

    // forward the transformer to get logits for the next token
    std::unique_ptr<float[]> logits = forward(transformer, token, pos, sampler->vocab_size);
    next = sample(sampler, logits.get());
    pos++;

    if (user_idx >= num_prompt_tokens && next != 128009 && next != 128001 && next != 128006) {
      // the Assistant is responding, so print its output
      char *piece = decode(tokenizer, token, next);
      safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
      fflush(stdout);
    }
    if ((next == 128009 || next == 128001) && user_idx >= num_prompt_tokens) {
      printf("\n");
    }
  }
  printf("\n");
  free(prompt_tokens);
  free(system_prompt_tokens);
  free(user_prompt_tokens);
  free(system_prompt);
  free(user_prompt);
}

// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

void error_usage() {
  fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
  fprintf(stderr, "Example: run model.bin -n 4096 -i \"Once upon a time\"\n");
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
  fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
  fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
  fprintf(stderr, "  -n <int>    number of steps to run for, default 4096. 0 = max_seq_len\n");
  fprintf(stderr, "  -i <string> input prompt\n");
  fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
  fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
  fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
  exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
  // default parameters
  char *checkpoint_path = NULL; // e.g. out/model.bin
  const char *tokenizer_path = "tokenizer.bin";
  float temperature = 1.0f;        // 0.0 = greedy deterministic. 1.0 = original. don't set higher
  float topp = 0.9f;               // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
  int steps = 4096;                // number of steps to run for
  char *prompt = NULL;             // prompt string
  unsigned long long rng_seed = 0; // seed rng with time by default
  const char *mode = "generate";         // generate|chat
  char *system_prompt = NULL;      // the (optional) system prompt to use in chat mode

  // poor man's C argparse so we can override the defaults above from the command line
  if (argc >= 2) {
    checkpoint_path = argv[1];
  } else {
    error_usage();
  }
  for (int i = 2; i < argc; i += 2) {
    // do some basic validation
    if (i + 1 >= argc) {
      error_usage();
    } // must have arg after flag
    if (argv[i][0] != '-') {
      error_usage();
    } // must start with dash
    if (strlen(argv[i]) != 2) {
      error_usage();
    } // must be -x (one dash, one letter)
    // read in the args
    if (argv[i][1] == 't') {
      temperature = atof(argv[i + 1]);
    } else if (argv[i][1] == 'p') {
      topp = atof(argv[i + 1]);
    } else if (argv[i][1] == 's') {
      rng_seed = atoi(argv[i + 1]);
    } else if (argv[i][1] == 'n') {
      steps = atoi(argv[i + 1]);
    } else if (argv[i][1] == 'i') {
      prompt = argv[i + 1];
    } else if (argv[i][1] == 'z') {
      tokenizer_path = argv[i + 1];
    } else if (argv[i][1] == 'm') {
      mode = argv[i + 1];
    } else if (argv[i][1] == 'y') {
      system_prompt = argv[i + 1];
    } else {
      error_usage();
    }
  }

  // parameter validation/overrides
  if (rng_seed <= 0)
    rng_seed = (unsigned int)time(NULL);
  if (temperature < 0.0f)
    temperature = 0.0f;
  if (topp < 0.0f || 1.0f < topp)
    topp = 0.9f;
  if (steps < 0)
    steps = 0;

  // Initialize HIP stream for async operations
  HIP_CHECK(hipStreamCreate(&g_stream));

  // build the Transformer via the model .bin file (returns device pointer, config, and weights pointer)
  auto [transformer, config, weights_ptr] = build_transformer(checkpoint_path);

  if (steps == 0 || steps > config.seq_len)
    steps = config.seq_len; // override to ~max length

  // build the Tokenizer via the tokenizer .bin file
  Tokenizer tokenizer;
  build_tokenizer(&tokenizer, tokenizer_path, config.vocab_size);

  // build the Sampler
  Sampler sampler;
  build_sampler(&sampler, config.vocab_size, temperature, topp, rng_seed);

  // run!
  if (strcmp(mode, "generate") == 0) {
    generate(transformer, &tokenizer, &sampler, prompt, steps);
  } else if (strcmp(mode, "chat") == 0) {
    chat(transformer, &tokenizer, &sampler, prompt, system_prompt, steps);
  } else {
    fprintf(stderr, "unknown mode: %s\n", mode);
    error_usage();
  }

  // memory and file handles cleanup
  free_sampler(&sampler);
  free_tokenizer(&tokenizer);
  free_transformer(transformer, weights_ptr);

  // Destroy HIP stream
  HIP_CHECK(hipStreamDestroy(g_stream));

  return 0;
}
#endif
