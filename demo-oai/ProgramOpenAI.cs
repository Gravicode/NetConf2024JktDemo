using System.ClientModel;
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

internal class ProgramOpenAI
{
    
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
        await DemoMultimodal();

    }
    
    static async Task DemoBasicPrompt()
    {      

        // Configure AI service credentials used by the kernel
        var (useAzureOpenAI, model, azureEndpoint, apiKey, orgId) = Settings.LoadFromFile();

        IChatClient client =
            new OpenAIClient(apiKey)
                .AsChatClient("gpt-4o-mini");

        Console.WriteLine(await client.CompleteAsync("Apa itu AI?"));
    }
    static async Task DemoChatHistory()
    {

        // Configure AI service credentials used by the kernel
        var (useAzureOpenAI, model, azureEndpoint, apiKey, orgId) = Settings.LoadFromFile();

        IChatClient client =
            new OpenAIClient(apiKey)
                .AsChatClient("gpt-4o-mini");
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

        // Configure AI service credentials used by the kernel
        var (useAzureOpenAI, model, azureEndpoint, apiKey, orgId) = Settings.LoadFromFile();

        IChatClient client =
            new OpenAIClient(apiKey)
                .AsChatClient("gpt-4o-mini");
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
        // Configure AI service credentials used by the kernel
        var (useAzureOpenAI, model, azureEndpoint, apiKey, orgId) = Settings.LoadFromFile();

        IChatClient openaiClient =
            new OpenAIClient(apiKey)
                .AsChatClient("gpt-4o-mini");

        var client = new ChatClientBuilder(openaiClient)
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
        // Configure AI service credentials used by the kernel
        var (useAzureOpenAI, model, azureEndpoint, apiKey, orgId) = Settings.LoadFromFile();

        IDistributedCache cache = new MemoryDistributedCache(Options.Create(new MemoryDistributedCacheOptions()));

        IChatClient openaiClient =
            new OpenAIClient(apiKey)
                .AsChatClient("gpt-4o-mini");

        IChatClient client = new ChatClientBuilder(openaiClient)
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
        // Configure AI service credentials used by the kernel
        var (useAzureOpenAI, model, azureEndpoint, apiKey, orgId) = Settings.LoadFromFile();

        // Configure OpenTelemetry exporter
        var sourceName = Guid.NewGuid().ToString();
        var tracerProvider = OpenTelemetry.Sdk.CreateTracerProviderBuilder()
            .AddSource(sourceName)
            .AddConsoleExporter()
            .Build();

        IChatClient openaiClient =
            new OpenAIClient(apiKey)
                .AsChatClient("gpt-4o-mini");
        using var loggerFactory = LoggerFactory.Create(builder =>
        {
            builder.AddConsole().SetMinimumLevel(LogLevel.Information); // Set the minimum log level
        });
        IChatClient client = new ChatClientBuilder(openaiClient)
        .UseOpenTelemetry(loggerFactory, sourceName, c => c.EnableSensitiveData = true)
        .Build();
        Console.ForegroundColor = ConsoleColor.Red;
        Console.WriteLine(await client.CompleteAsync("Apa itu AI?"));
    }

    static async Task DemoEmbedding()
    {// Configure AI service credentials used by the kernel
        var (useAzureOpenAI, model, azureEndpoint, apiKey, orgId) = Settings.LoadFromFile();
        IEmbeddingGenerator<string, Embedding<float>> generator =
    new OpenAIClient(apiKey)
        .AsEmbeddingGenerator("text-embedding-3-small");

        var embeddings = await generator.GenerateAsync(["Berapa luas Indonesia?"]);

        Console.WriteLine(string.Join(", ", embeddings[0].Vector.ToArray()));
    }
    
    static async Task DemoEmbeddingCaching()
    {// Configure AI service credentials used by the kernel
        var (useAzureOpenAI, model, azureEndpoint, apiKey, orgId) = Settings.LoadFromFile();
        IDistributedCache cache = new MemoryDistributedCache(Options.Create(new MemoryDistributedCacheOptions()));

        IEmbeddingGenerator<string, Embedding<float>> openAIGenerator =
            new OpenAIClient(apiKey)
                .AsEmbeddingGenerator("text-embedding-3-small");

        IEmbeddingGenerator<string, Embedding<float>> generator = new EmbeddingGeneratorBuilder<string, Embedding<float>>(openAIGenerator)
            .UseDistributedCache(cache)
            .Build();

        foreach (var prompt in new[] { "Apa itu AI?", "Apa itu AI?", "Apa itu AI?" })
        {
            var embeddings = await generator.GenerateAsync([prompt]);

            Console.WriteLine(string.Join(", ", embeddings[0].Vector.ToArray()));
        }
    }    
    static async Task DemoDI()
    {// Configure AI service credentials used by the kernel
        var (useAzureOpenAI, model, azureEndpoint, apiKey, orgId) = Settings.LoadFromFile();
        // App Setup
        var builder = Host.CreateApplicationBuilder();
        builder.Services.AddSingleton(new OpenAIClient(apiKey));
        builder.Services.AddDistributedMemoryCache();
        builder.Services.AddLogging(b => b.AddConsole().SetMinimumLevel(LogLevel.Trace));        
        builder.Services.AddChatClient(services => services.GetRequiredService<OpenAIClient>().AsChatClient("gpt-4o-mini"))
            .UseDistributedCache()
            .UseRetryOnRateLimit()
            .UseLanguage("Sunda")
            .UseLogging();

        var app = builder.Build();

        // Elsewhere in the app
        var chatClient = app.Services.GetRequiredService<IChatClient>();
        Console.WriteLine(await chatClient.CompleteAsync("Siapa wanita tercantik di indonesia?"));
        Console.WriteLine(await chatClient.CompleteAsync("Siapa wanita tertinggi di indonesia?"));
        Console.WriteLine(await chatClient.CompleteAsync("Siapa laki laki terganteng di indonesia?"));
        Console.WriteLine(await chatClient.CompleteAsync("Siapa laki laki tertinggi di indonesia?"));
    }
    static async Task DemoRAG()
    {// Configure AI service credentials used by the kernel
        var (useAzureOpenAI, model, azureEndpoint, apiKey, orgId) = Settings.LoadFromFile();
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
   new OpenAIClient(apiKey)
       .AsEmbeddingGenerator("text-embedding-3-small");
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

    static async Task DemoMultimodal()
    {

        // Configure AI service credentials used by the kernel
        var (useAzureOpenAI, model, azureEndpoint, apiKey, orgId) = Settings.LoadFromFile();

        IChatClient client =
            new OpenAIClient(apiKey)
                .AsChatClient("gpt-4o-mini");
        var raiseAlert = AIFunctionFactory.Create((string Name, string alertReason) =>
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("*** GENDER ALERT ***");
            Console.WriteLine($"Name: {Name}: {alertReason}");
            Console.ForegroundColor = ConsoleColor.White;
        }, "RaiseAlert");
        var chatOptions = new ChatOptions { Tools = [raiseAlert] };
        //Console.WriteLine(await client.CompleteAsync("Apa itu AI?"));
        var files = Directory.GetFiles(@"../../../KTP","*.jpg");
        foreach (var imagePath in files)
        {
          
            var message = new ChatMessage(ChatRole.User, $$"""
        Extract information from this identity card image. Raise an alert only if the gender is female,
        """);
            message.Contents.Add(new ImageContent(File.ReadAllBytes(imagePath), "image/jpg"));
            var response = await client.CompleteAsync<PersonInfo>([message], chatOptions);//useNativeJsonSchema: isOllama

            if (response.TryGetResult(out var result))
            {
                Console.WriteLine(result);
                Console.WriteLine("------------------------");
            }
        }
    }
}
public class PersonInfo
{
    public string KTPNumber { get; set; }
    public string Name { get; set; }
    public string Address { get; set; }
    public string Job { get; set; }
    public Genders Gender { get; set; }
    public string BirthPlace { get; set; }
    public DateTime BirthDate { get; set; }
    public DateTime KTPDate { get; set; }
    public bool Married { get; set; }

    public override string ToString()
    {
        return $"Name:{Name}\nKTP No:{KTPNumber}\nJob:{Job}\nGender:{Gender}\nBirth Place:{BirthPlace}\nBirth Date:{BirthDate}\nKTP Date:{KTPDate}\nMarried:{(Married ? "Ya":"Tidak")}";
    }

    public enum Genders { Female, Male, Unknown }
}
public class Movie
{
    [VectorStoreRecordKey]
    public int Key { get; set; }

    [VectorStoreRecordData]
    public string Title { get; set; }

    [VectorStoreRecordData]
    public string Description { get; set; }

    [VectorStoreRecordVector(384, DistanceFunction.CosineSimilarity)]
    public ReadOnlyMemory<float> Vector { get; set; }
}

public static class RetryOnRateLimitExtensions
{
    public static ChatClientBuilder UseRetryOnRateLimit(this ChatClientBuilder builder)
        => builder.Use(next => new RetryOnRateLimitChatClient(next));

    private class RetryOnRateLimitChatClient(IChatClient innerClient) : DelegatingChatClient(innerClient)
    {
        public override async Task<ChatCompletion> CompleteAsync(IList<ChatMessage> chatMessages, ChatOptions? options = null, CancellationToken cancellationToken = default)
        {
            while (true)
            {
                try
                {
                    return await base.CompleteAsync(chatMessages, options, cancellationToken);
                }
                catch (ClientResultException ex) when (ex.Message.Contains("HTTP 429"))
                {
                    Console.WriteLine("Rate limited exceeded. Retrying in 3 seconds");
                    await Task.Delay(TimeSpan.FromSeconds(3), cancellationToken);
                }
            }
        }
    }
}

public static class UseLanguageExtensions
{
    public static ChatClientBuilder UseLanguage(this ChatClientBuilder builder,string Bahasa="Arab")
        => builder.Use(next => new UseLanguageChatClient(next, Bahasa));

    private class UseLanguageChatClient(IChatClient innerClient,string Bahasa) : DelegatingChatClient(innerClient)
    {
        public override async Task<ChatCompletion> CompleteAsync(IList<ChatMessage> chatMessages, ChatOptions? options = null, CancellationToken cancellationToken = default)
        {
            var newMsg = new ChatMessage(ChatRole.User, $"Jawab selalu dalam bahasa {Bahasa}.");
            try
            {
                chatMessages.Add(newMsg);
                return await base.CompleteAsync(chatMessages, options, cancellationToken);
            }
            finally
            {
                chatMessages.Remove(newMsg);
                Console.WriteLine("Hitting use language middleware");
                
            }
        }
    }
}