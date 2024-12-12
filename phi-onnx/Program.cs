using System.Text;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.Onnx;
#pragma warning disable SKEXP0070 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.

internal class Program
{
    private static async Task Main(string[] args)
    {
        StringBuilder chatPrompt = new("""
                               <message role="system">You are a librarian, expert about books</message>
                               <message role="user">Hi, I'm looking for book suggestions</message>
                               """);

        Console.WriteLine("======== Onnx - Chat Completion Streaming ========");

        var kernel = Kernel.CreateBuilder()
            .AddOnnxRuntimeGenAIChatCompletion(
                modelId: "phi3-mini-4k-instruct-cpu-int4-rtn-block-32-acc-level-4",
                modelPath: @"C:\Users\mifma\Downloads\Phi-3-mini-4k-instruct-onnx")
            .Build();

        var reply = await StreamMessageOutputFromKernelAsync(kernel, chatPrompt.ToString());

        chatPrompt.AppendLine($"<message role=\"assistant\"><![CDATA[{reply}]]></message>");
        chatPrompt.AppendLine("<message role=\"user\">I love history and philosophy, I'd like to learn something new about Greece, any suggestion</message>");

        reply = await StreamMessageOutputFromKernelAsync(kernel, chatPrompt.ToString());

        Console.WriteLine(reply);

        DisposeServices(kernel);
    }

    private static async Task StreamMessageOutputAsync(OnnxRuntimeGenAIChatCompletionService chatCompletionService, ChatHistory chatHistory, AuthorRole authorRole)
    {
        bool roleWritten = false;
        string fullMessage = string.Empty;

        await foreach (var chatUpdate in chatCompletionService.GetStreamingChatMessageContentsAsync(chatHistory))
        {
            if (!roleWritten && chatUpdate.Role.HasValue)
            {
                Console.Write($"{chatUpdate.Role.Value}: {chatUpdate.Content}");
                roleWritten = true;
            }

            if (chatUpdate.Content is { Length: > 0 })
            {
                fullMessage += chatUpdate.Content;
                Console.Write(chatUpdate.Content);
            }
        }

        Console.WriteLine("\n------------------------");
        chatHistory.AddMessage(authorRole, fullMessage);
    }

    private static async Task<string> StreamMessageOutputFromKernelAsync(Kernel kernel, string prompt)
    {
        bool roleWritten = false;
        string fullMessage = string.Empty;

        await foreach (var chatUpdate in kernel.InvokePromptStreamingAsync<StreamingChatMessageContent>(prompt))
        {
            if (!roleWritten && chatUpdate.Role.HasValue)
            {
                Console.Write($"{chatUpdate.Role.Value}: {chatUpdate.Content}");
                roleWritten = true;
            }

            if (chatUpdate.Content is { Length: > 0 })
            {
                fullMessage += chatUpdate.Content;
                Console.Write(chatUpdate.Content);
            }
        }

        Console.WriteLine("\n------------------------");
        return fullMessage;
    }

    /// <summary>
    /// To avoid any potential memory leak all disposable services created by the kernel are disposed.
    /// </summary>
    /// <param name="kernel">Target kernel</param>
    private static void DisposeServices(Kernel kernel)
    {
        foreach (var target in kernel
            .GetAllServices<IChatCompletionService>()
            .OfType<IDisposable>())
        {
            target.Dispose();
        }
    }
}
#pragma warning restore SKEXP0070 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.
