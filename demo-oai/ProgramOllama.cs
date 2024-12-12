using System.ComponentModel;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Caching.Distributed;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Microsoft.Extensions.VectorData;
using Microsoft.SemanticKernel.Connectors.InMemory;
using OpenAI;
using OpenTelemetry.Trace;

internal class ProgramOllama
{
    const string OllamaEndpoint = "http://localhost:11434/";
    const string ChatModel = "llama3.2:3b";
    const string EmbeddingModel = "all-minilm";

   /*
    private static async Task Main(string[] args)
    {
        await DemoBasicPrompt();
        await DemoChatHistory();
        await DemoChatStreaming();
        await DemoFunctionCalling();
        await DemoCaching();
        await DemoTelemetry();
        await DemoEmbedding();
        await DemoEmbeddingCaching();
        await DemoDI();
        await DemoRAG();

    }*/
   
    static async Task DemoBasicPrompt()
    {   
        IChatClient client = new OllamaChatClient(OllamaEndpoint, ChatModel);

        Console.WriteLine(await client.CompleteAsync("Apa itu AI?"));
    }
    static async Task DemoChatHistory()
    {
        IChatClient client = new OllamaChatClient(OllamaEndpoint, ChatModel);
        List<ChatMessage> chatHistory = [
    new ChatMessage(ChatRole.System, "Kamu ada virtual asisten perempuan yang ramah dan lucu."),
    new ChatMessage(ChatRole.User, "Apa profesi Kang Memet"),
    new ChatMessage(ChatRole.Assistant, "Tukang Ketoprak"),
    new ChatMessage(ChatRole.User, "Dimana rumah Kang Memet"),
    new ChatMessage(ChatRole.Assistant, "Di Gang Jengkol No. 21"),
];
        Console.ForegroundColor = ConsoleColor.Green;
        Console.Write("Tanya:");
        var tanya = Console.ReadLine();
        chatHistory.Add(new ChatMessage(ChatRole.Assistant, tanya));
        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.Write("Jawab:");
        Console.WriteLine(await client.CompleteAsync(chatHistory));
    }
    static async Task DemoChatStreaming()
    {

        IChatClient client = new OllamaChatClient(OllamaEndpoint, ChatModel);
        List<ChatMessage> chatHistory = [
    new ChatMessage(ChatRole.System, "Kamu ada virtual asisten perempuan yang romantis, ramah dan lucu. Nama kamu Sheila."),    
];
        Console.WriteLine("Ketik 'q' untuk exit !!");
        while (true)
        {
            Console.ForegroundColor = ConsoleColor.Green;
            Console.Write("Tanya: ");
            var tanya = Console.ReadLine();
            if (tanya.Equals("q", StringComparison.InvariantCultureIgnoreCase)) break;
            chatHistory.Add(new ChatMessage(ChatRole.Assistant, tanya));
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.Write("Jawab: ");
            var jawaban = string.Empty;
            await foreach (var update in client.CompleteStreamingAsync(chatHistory))
            {
                Console.Write(update);
                jawaban += update;
            }
            chatHistory.Add(new ChatMessage(ChatRole.Assistant, jawaban));
            Console.WriteLine();
        }
    }
    static async Task DemoFunctionCalling()
    {
        IChatClient ollamaClient = new OllamaChatClient(OllamaEndpoint, ChatModel);

        var client = new ChatClientBuilder(ollamaClient)
            .UseFunctionInvocation()
            .Build();

        ChatOptions chatOptions = new()
        {
            Tools = [AIFunctionFactory.Create(GetWeather)]
        };

        await foreach (var message in client.CompleteStreamingAsync("Do I need an umbrella?", chatOptions))
        {
            Console.Write(message);
        }

        [Description("Gets the weather")]
        static string GetWeather()
        {
            var random = Random.Shared.NextDouble();
            var res =  random > 0.5 ? "It's sunny" : "It's raining";
            Console.WriteLine($"Hasil Dari Function => {random}: {res}");
            return res;
        }
    }
    static async Task DemoCaching()
    {
        

        IDistributedCache cache = new MemoryDistributedCache(Options.Create(new MemoryDistributedCacheOptions()));

        IChatClient ollamaClient = new OllamaChatClient(OllamaEndpoint, ChatModel);

        IChatClient client = new ChatClientBuilder(ollamaClient)
            .UseDistributedCache(cache)
            .Build();

        for (int i = 0; i < 3; i++)
        {
            await foreach (var message in client.CompleteStreamingAsync("Jelaskan dalam 100 karakter, apa itu ganteng?"))
            {
                Console.Write(message);
            }

            Console.WriteLine();
            Console.WriteLine();
        }
    }
    static async Task DemoTelemetry()
    {
        
        // Configure OpenTelemetry exporter
        var sourceName = Guid.NewGuid().ToString();
        var tracerProvider = OpenTelemetry.Sdk.CreateTracerProviderBuilder()
            .AddSource(sourceName)
            .AddConsoleExporter()
            .Build();

        IChatClient ollamaClient = new OllamaChatClient(OllamaEndpoint, ChatModel);
        using var loggerFactory = LoggerFactory.Create(builder =>
        {
            builder.AddConsole().SetMinimumLevel(LogLevel.Information); // Set the minimum log level
        });
        IChatClient client = new ChatClientBuilder(ollamaClient)
        .UseOpenTelemetry(loggerFactory, sourceName, c => c.EnableSensitiveData = true)
        .Build();
        Console.ForegroundColor = ConsoleColor.Red;
        Console.WriteLine(await client.CompleteAsync("Apa itu AI?"));
    }

    static async Task DemoEmbedding()
    {  
        IEmbeddingGenerator<string, Embedding<float>> generator =
    new OllamaEmbeddingGenerator(new Uri(OllamaEndpoint), EmbeddingModel); 

        var embeddings = await generator.GenerateAsync(["Berapa luas Indonesia?"]);

        Console.WriteLine(string.Join(", ", embeddings[0].Vector.ToArray()));
    }
    
    static async Task DemoEmbeddingCaching()
    {
        IDistributedCache cache = new MemoryDistributedCache(Options.Create(new MemoryDistributedCacheOptions()));

        IEmbeddingGenerator<string, Embedding<float>> ollamaGenerator =
            new OllamaEmbeddingGenerator(new Uri(OllamaEndpoint), EmbeddingModel);

        IEmbeddingGenerator<string, Embedding<float>> generator = new EmbeddingGeneratorBuilder<string, Embedding<float>>(ollamaGenerator)
            .UseDistributedCache(cache)
            .Build();

        foreach (var prompt in new[] { "Apa itu AI?", "Apa itu AI?", "Apa itu AI?" })
        {
            var embeddings = await generator.GenerateAsync([prompt]);

            Console.WriteLine(string.Join(", ", embeddings[0].Vector.ToArray()));
        }
    }    
    static async Task DemoDI()
    {
        // App Setup
        var builder = Host.CreateApplicationBuilder();
        IChatClient ollamaClient = new OllamaChatClient(OllamaEndpoint, ChatModel);
        builder.Services.AddSingleton(ollamaClient);
        builder.Services.AddDistributedMemoryCache();
        builder.Services.AddLogging(b => b.AddConsole().SetMinimumLevel(LogLevel.Trace));

        builder.Services.AddChatClient(services => services.GetRequiredService<OllamaChatClient>())
            .UseDistributedCache()
            .UseLogging();

        var app = builder.Build();

        // Elsewhere in the app
        var chatClient = app.Services.GetRequiredService<IChatClient>();
        Console.WriteLine(await chatClient.CompleteAsync("Siapa wanita tercantik di indonesia?"));
    }
    static async Task DemoRAG()
    {
        var movieData = new List<Movie>()
{
    new Movie
        {
            Key=0,
            Title="Lion King",
            Description="The Lion King is a classic Disney animated film that tells the story of a young lion named Simba who embarks on a journey to reclaim his throne as the king of the Pride Lands after the tragic death of his father."
        },
    new Movie
        {
            Key=1,
            Title="Inception",
            Description="Inception is a science fiction film directed by Christopher Nolan that follows a group of thieves who enter the dreams of their targets to steal information."
        },
    new Movie
        {
            Key=2,
            Title="The Matrix",
            Description="The Matrix is a science fiction film directed by the Wachowskis that follows a computer hacker named Neo who discovers that the world he lives in is a simulated reality created by machines."
        },
    new Movie
        {
            Key=3,
            Title="Shrek",
            Description="Shrek is an animated film that tells the story of an ogre named Shrek who embarks on a quest to rescue Princess Fiona from a dragon and bring her back to the kingdom of Duloc."
        }
};
        //put vector in memory
        var vectorStore = new InMemoryVectorStore();

        var movies = vectorStore.GetCollection<int, Movie>("movies");
        //create collection if not exist
        await movies.CreateCollectionIfNotExistsAsync();

        //create embedding generator
        IEmbeddingGenerator<string, Embedding<float>> generator =
   new OllamaEmbeddingGenerator(new Uri(OllamaEndpoint), EmbeddingModel); 
       
        //generate embedding from desc and update to vector field
        foreach (var movie in movieData)
        {
            movie.Vector = await generator.GenerateEmbeddingVectorAsync(movie.Description);
            await movies.UpsertAsync(movie);
        }

        //query data
        var query = "A family friendly movie";
        var queryEmbedding = await generator.GenerateEmbeddingVectorAsync(query);

        //set search options
        var searchOptions = new VectorSearchOptions()
        {
            Top = 1,
            VectorPropertyName = "Vector"
        };

        var results = await movies.VectorizedSearchAsync(queryEmbedding, searchOptions);
        //show result
        await foreach (var result in results.Results)
        {
            Console.WriteLine($"Title: {result.Record.Title}");
            Console.WriteLine($"Description: {result.Record.Description}");
            Console.WriteLine($"Score: {result.Score}");
            Console.WriteLine();
        }
    }
}