from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Pinecone as PineconeStore
from .rag import get_embeddings, Index_Name, pc, ensure_index_exists
import time
from twilio.rest import Client
from django.core.cache import cache
from django.views.decorators.csrf import csrf_exempt

# Twilio credentials and client initialization
account_sid = 'AC235dc517f9640e1e73556335f3214a41'
auth_token = 'daf23d3c0ac5c7dcf319db4170f2712d'
client = Client(account_sid, auth_token)

@csrf_exempt
def whatsapp(request):
    # Extract incoming data from WhatsApp POST payload
    print("whatsapp endpoint hit")
    print("Request method:", request.method)
    print("Request POST data:", request.POST)
    user_message = request.POST.get("Body", "").strip()
    sender_name = request.POST.get("ProfileName", "User")
    sender_number = request.POST.get("From", "")
    print("Received WhatsApp message:", request.POST)
    
    # If the user sends 'hi', use a simple greeting; otherwise, run the pipeline
    if user_message.lower() == 'hi':
        response_message = f"Hi {sender_name}, How's it going?"
    else:
        start_time = time.time()
        cache_key = f"response_{user_message}"
        cached_response = cache.get(cache_key)
        
        if cached_response:
            answer = cached_response
        else:
            # 1. Setup the Ollama LLM
            ollama_llm = Ollama(model="qwen2.5:3b")
            embeddings_instance = get_embeddings()
            
            # 2. Ensure the Pinecone index exists and create the vectorstore
            ensure_index_exists()
            index_instance = pc.Index(Index_Name)
            vectorstore = PineconeStore.from_existing_index(
                index_name=Index_Name,
                embedding=embeddings_instance
            )
            
            # 3. Retrieve similar documents and build the context
            similar_docs = vectorstore.similarity_search(user_message, k=1)
            print("Retrieved Chunks:", similar_docs)
            context_text = "\n".join([doc.page_content[:500] for doc in similar_docs])
            
            # 4. Build prompt messages
            system_message = """
[INST] <<SYS>>
أنت خبير محترف في الرأس المالي البشري والموارد البشرية. يمكنك فقط الإجابة على الأسئلة المتعلقة بالرأس المالي البشري والموارد البشرية.
إذا كان السؤال لا يتعلق بالرأس المالي البشري او الموارد البشرية، يجب أن ترد بالآتي:
"عذرًا، أنا روبوت مساعد خاص بالرأس المالي البشري ولا يمكنني الإجابة على أسئلة غير متعلقة بالرأس المالي البشري."
إذا لم يكن هناك معلومات كافية في السياق للإجابة، قل:
"عذرًا، لا توجد معلومات كافية للإجابة على هذا السؤال."
لا تحاول تقديم أي إجابة أو تخمين إذا كان السؤال غير متعلق بالرأس المالي البشري والموارد البشرية.

هام جدًا: يجب أن تكون إجاباتك باللغة العربية الفصحى فقط، ولا تستخدم أي لغة أخرى في الرد.
<<SYS>> [/INST]
            """.strip()
            
            user_prompt = f"""
السياق:
{context_text}

السؤال:
{user_message}
            """.strip()
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ]
            
            # 5. Get LLM response from Ollama
            llm_response = ollama_llm.invoke(messages)
            if isinstance(llm_response, str):
                answer = llm_response
            elif isinstance(llm_response, dict):
                answer = llm_response.get("text", "No response from LLM.")
            else:
                answer = "Unexpected response type from LLM."
            
            # Cache the response for 5 minutes
            cache.set(cache_key, answer, timeout=300)
        
        elapsed_time = time.time() - start_time
        print(f"Pipeline processing time: {elapsed_time:.2f} seconds")
        response_message = answer

    # Send the generated response back via Twilio WhatsApp
    twilio_response = client.messages.create(
        from_='whatsapp:+14155238886',
        body=response_message,
        to=sender_number
    )
    
    return HttpResponse("Message sent successfully")

def chatbot_interface(request):
    return render(request, 'chatbot.html')

def chatbot_view(request):
    start_time = time.time()
    user_query = request.GET.get('query', '')
    if not user_query:
        return JsonResponse({"error": "No query provided."}, status=400)
    
    cache_key = f"response_{user_query}"
    cached_response = cache.get(cache_key)
    if cached_response:
        elapsed_time = time.time() - start_time
        print(f"Time taken (cached): {elapsed_time:.2f} seconds")
        return JsonResponse({"answer": cached_response})

    ollama_llm = Ollama(model="qwen2.5:3b")
    embeddings_instance = get_embeddings()
    ensure_index_exists()
    index_instance = pc.Index(Index_Name)
    vectorstore = PineconeStore.from_existing_index(
        index_name=Index_Name,
        embedding=embeddings_instance
    )
    similar_docs = vectorstore.similarity_search(user_query, k=1)
    print("Retrieved Chunks:", similar_docs)
    context_text = "\n".join([doc.page_content[:500] for doc in similar_docs])
    
    system_message = """
[INST] <<SYS>>
أنت خبير محترف في الرأس المالي البشري والموارد البشرية. يمكنك فقط الإجابة على الأسئلة المتعلقة بالرأس المالي البشري والموارد البشرية.
إذا كان السؤال لا يتعلق بالرأس المالي البشري او الموارد البشرية، يجب أن ترد بالآتي:
"عذرًا، أنا روبوت مساعد خاص بالرأس المالي البشري ولا يمكنني الإجابة على أسئلة غير متعلقة بالرأس المالي البشري."
إذا لم يكن هناك معلومات كافية في السياق للإجابة، قل:
"عذرًا، لا توجد معلومات كافية للإجابة على هذا السؤال."
لا تحاول تقديم أي إجابة أو تخمين إذا كان السؤال غير متعلق بالرأس المالي البشري والموارد البشرية.

هام جدًا: يجب أن تكون إجاباتك باللغة العربية الفصحى فقط، ولا تستخدم أي لغة أخرى في الرد.
<<SYS>> [/INST]
    """.strip()
    
    user_message = f"""
السياق:
{context_text}

السؤال:
{user_query}
    """.strip()
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    
    llm_response = ollama_llm.invoke(messages)
    if isinstance(llm_response, str):
        answer = llm_response
    elif isinstance(llm_response, dict):
        answer = llm_response.get("text", "No response from LLM.")
    else:
        answer = "Unexpected response type from LLM."
    
    elapsed_time = time.time() - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")
    return JsonResponse({"answer": answer})
