import subprocess
import sys
import pytest
import grpc
from text_embeddings_server.pb import embed_pb2, embed_pb2_grpc
import transformers


# @pytest.fixture(scope="session")
# def grpc_server():
#     # Specify the command to run the server
#     server_process = subprocess.Popen(
#         [sys.executable, 'path/to/cli.py', 'serve', '--model_path', 'path/to/model', '--uds_path', '/tmp/text-embeddings-server'],
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE,
#         text=True,  # Ensure outputs are in string format
#         bufsize=1,  # Line-buffered
#         universal_newlines=True
#     )

#     # Loop to check stdout for a specific message
#     print("Waiting for the server to start...")
#     for line in server_process.stdout:
#         print("Server output:", line.strip())  # Optional: to see the server outputs in pytest output
#         if "Server started" in line:
#             print("Server has started.")
#             break

#     yield  # Yield control back to pytest for running the actual tests

#     # Terminate the server after tests are done
#     server_process.terminate()
#     server_process.wait()

@pytest.fixture
def grpc_channel():
  # Create a gRPC channel to connect to the server
  channel = grpc.insecure_channel("unix:///tmp/text-embeddings-server")
  yield channel
  # Teardown: close the channel
  channel.close()

@pytest.fixture
def tokenizer():
  # Load the tokenizer
  tokenizer = transformers.AutoTokenizer.from_pretrained("BAAI/bge-small-en")
  return tokenizer

def test_grpc_post_call(grpc_channel, tokenizer):
  # Create a gRPC stub for the server
  stub = embed_pb2_grpc.EmbeddingServiceStub(grpc_channel)
  
  input_text = "Hello, world!"
  tokens = tokenizer(input_text)
  
  # Make the gRPC POST call
  request = embed_pb2.EmbedRequest(input_ids=tokens["input_ids"],token_type_ids=tokens["token_type_ids"],cu_seq_lengths=[6],max_length=len(tokens["input_ids"]))
  response = stub.Embed(request)
  # Assert the response
  assert len(response.embeddings) > 0  