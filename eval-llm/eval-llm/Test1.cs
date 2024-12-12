using Microsoft.Extensions.AI;
using Microsoft.Extensions.AI.Evaluation;
using Microsoft.Extensions.AI.Evaluation.Quality;
using Microsoft.Extensions.AI.Evaluation.Reporting;
using Microsoft.Extensions.AI.Evaluation.Reporting.Storage.Disk;
using Microsoft.ML.Tokenizers;
using OpenAI;
using System.Text;
using System.Text.Json;

namespace eval_llm
{
    [TestClass]
    public sealed class Test1
    {
      IList<ChatMessage> chats { set; get; }
      IChatClient currentClient{ set; get; }
        public Test1()
        {
            if (currentClient == null)
            {
                currentClient =
                    new OpenAIClient(AppConstants.OpenAIKey)
                        .AsChatClient(AppConstants.Model);
            }



        }
        [TestMethod]
        public async Task EvalTest1()
        {
           
            // This is the example question and answer that will be evaluated.
            var question = new EvalQuestion
            {
                QuestionId = 1,
                Question = "Apa ibu kota indonesia?",
                Answer = "DKI Jakarta"
            };
            chats =
[
  new ChatMessage(ChatRole.System, "You are a helpful AI assistant") 
];
            // Construct a reporting configuration to support the evaluation
            var reportingConfiguration = GetReportingConfiguration();

            // Run an evaluation pass and record the results to the cache folder
            await EvaluateQuestion(question, reportingConfiguration, CancellationToken.None);
        }

        [TestMethod]
        public async Task EvalTest2()
        {
            chats =
[
    new ChatMessage(ChatRole.System, "You are a helpful AI assistant."),
    //new ChatMessage(ChatRole.User, "Dimana kantor gravicode?"),
    //new ChatMessage(ChatRole.Assistant, "Bogor"),
];
            
            // This is the example question and answer that will be evaluated.
            var question = new EvalQuestion
            {
                QuestionId = 2,
                Question = "Dimana kantor gravicode?",
                Answer = "Bogor"
            };

            // Construct a reporting configuration to support the evaluation
            var reportingConfiguration = GetReportingConfiguration();

            // Run an evaluation pass and record the results to the cache folder
            await EvaluateQuestion(question, reportingConfiguration, CancellationToken.None);
        }
        private async Task EvaluateQuestion(EvalQuestion question, ReportingConfiguration reportingConfiguration, CancellationToken cancellationToken)
        {
            
            
            // Create a new ScenarioRun to represent each evaluation run.
            await using ScenarioRun scenario = await reportingConfiguration.CreateScenarioRunAsync($"Question_{question.QuestionId}", cancellationToken: cancellationToken);

            // Run the sample through the assistant to generate a response.
            chats.Add(new ChatMessage(ChatRole.User, question.Question));
            var responseItems = await currentClient.CompleteAsync(chats);
            chats.Add(new ChatMessage(ChatRole.Assistant, responseItems.ToString()));
            var answerBuilder = new StringBuilder();
          
            answerBuilder.Append(responseItems.ToString());
               
            var finalAnswer = answerBuilder.ToString();

            // Invoke the evaluators
            EvaluationResult evalResult = await scenario.EvaluateAsync(
                [new ChatMessage(ChatRole.User, question.Question)],
                new ChatMessage(ChatRole.Assistant, finalAnswer),
                additionalContext: [new AnswerScoringEvaluator.Context(question.Answer)],
                cancellationToken);

            // Assert that the evaluator was able to successfully generate an analysis
            Assert.IsFalse(evalResult.Metrics.Values.Any(m => m.Interpretation?.Rating == EvaluationRating.Inconclusive), "Model response was inconclusive");

            // Assert that the evaluators did not report any diagnostic errors
            Assert.IsFalse(evalResult.ContainsDiagnostics(d => d.Severity == EvaluationDiagnosticSeverity.Error), "Evaluation had errors.");

        }
        ReportingConfiguration GetReportingConfiguration()
        {
            // Setup and configure the evaluators you would like to utilize for each AI chat.
            // AnswerScoringEvaluator is an example of a custom evaluator that can be added, while the others
            // are included in the evaluation library.

            // Measures the extent to which the model's generated responses are pertinent and directly related to the given queries.
            IEvaluator rtcEvaluator =
                new RelevanceTruthAndCompletenessEvaluator(
                    new RelevanceTruthAndCompletenessEvaluator.Options(includeReasoning: true));
            // Measures how well the language model can produce output that flows smoothly, reads naturally, and resembles human-like language.
            IEvaluator coherenceEvaluator = new CoherenceEvaluator();
            // Measures the grammatical proficiency of a generative AI's predicted answer.
            IEvaluator fluencyEvaluator = new FluencyEvaluator();
            // Measures how well the model's generated answers align with information from the source data
            IEvaluator groundednessEvaluator = new GroundednessEvaluator();
            // Measures the extent to which the model's retrieved documents are pertinent and directly related to the given queries.
            IEvaluator answerScoringEvaluator = new AnswerScoringEvaluator();
            
            IChatClient chatClient = new OpenAIClient(AppConstants.OpenAIKey)
        .AsChatClient(AppConstants.Model);
            // Setup the chat client that is used to perform the evaluations
            Tokenizer tokenizer = TiktokenTokenizer.CreateForModel("gpt-4o");

            var chatConfig = new ChatConfiguration(chatClient, tokenizer.ToTokenCounter(inputTokenLimit: 6000));
            var StorageRootPath = @"C:\Experiment\netconf2024demo\storage";
            var ExecutionName = "TestLLM";
            // The DiskBasedReportingConfiguration caches LLM responses to reduce costs and
            // increase test run performance.
            return DiskBasedReportingConfiguration.Create(
                    storageRootPath: StorageRootPath,
                    chatConfiguration: chatConfig,
                    evaluators: [
                        rtcEvaluator,
                coherenceEvaluator,
                fluencyEvaluator,
                groundednessEvaluator,
                answerScoringEvaluator],
                    executionName: ExecutionName);
        }
    }

    public sealed class AnswerScoringEvaluator : ChatConversationEvaluator
    {

        public sealed class Context(string expectedAnswer) : EvaluationContext
        {
            public string ExpectedAnswer { get; } = expectedAnswer;
        }

        const string MetricName = "Answer Score";

        protected override bool IgnoresHistory => true;

        public override IReadOnlyCollection<string> EvaluationMetricNames => [MetricName];

        protected override EvaluationResult InitializeResult()
        {
            return new EvaluationResult(new NumericMetric(MetricName));
        }

        protected override async ValueTask<string> RenderEvaluationPromptAsync(
            ChatMessage? userRequest,
            ChatMessage modelResponse,
            IEnumerable<ChatMessage>? includedHistory,
            IEnumerable<EvaluationContext>? additionalContext,
            CancellationToken token)
        {
            string renderedModelResponse = await this.RenderAsync(modelResponse, token);

            string renderedUserRequest =
                userRequest is not null
                    ? await this.RenderAsync(userRequest, token)
                    : string.Empty;

            string answer = "";

            if (additionalContext is not null &&
                additionalContext.OfType<Context>().FirstOrDefault() is Context context)
            {
                answer = context.ExpectedAnswer;
            }
            else
            {
                throw new InvalidOperationException($"The ExpectedAnswer must be provided in the additional context.");
            }

            List<string> scoreWords = ["Awful", "Poor", "Good", "Perfect"];

            var prompt = $$"""
        There is an AI assistant that answers questions about products sold by an online retailer. The questions
        may be asked by customers or by customer support agents.

        You are evaluating the quality of an AI assistant's response to several questions. Here are the
        questions, the desired true answers, and the answers given by the AI system:

        <questions>
            <question index="0">
                <text>{{renderedUserRequest}}</text>
                <truth>{{answer}}</truth>
                <assistantAnswer>{{renderedModelResponse}}</assistantAnswer>
            </question>
        </questions>

        Evaluate each of the assistant's answers separately by replying in this JSON format:

        {
            "scores": [
                { "index": 0, "descriptionOfQuality": string, "scoreLabel": number },
                { "index": 1, "descriptionOfQuality": string, "scoreLabel": number },
                ... etc ...
            ]
        ]

        Score only based on whether the assistant's answer is true and answers the question. As long as the
        answer covers the question and is consistent with the truth, it should score as perfect. There is
        no penalty for giving extra on-topic information or advice. Only penalize for missing necessary facts
        or being misleading.

        The descriptionOfQuality should be up to 5 words summarizing to what extent the assistant answer
        is correct and sufficient.

        Based on descriptionOfQuality, the scoreLabel must be a number between 1 and 5 inclusive, where 5 is best and 1 is worst.
        Do not use any other words for scoreLabel. You may only pick one of those scores.
        
        """
            ;

            return prompt;
        }

        protected override ValueTask ParseEvaluationResponseAsync(
            string modelResponseForEvaluationPrompt,
            EvaluationResult result,
            ChatConfiguration configuration,
            CancellationToken token)
        {
            bool hasMetric = result.TryGet<NumericMetric>(MetricName, out var numericMetric);
            if (!hasMetric || numericMetric is null)
            {
                throw new Exception("NumericMetric was not properly initialized.");
            }

            var jsonOptions = new JsonSerializerOptions(JsonSerializerDefaults.Web);

            var parsedResponse = JsonSerializer.Deserialize<ScoringResponse>(TrimMarkdownDelimiters(modelResponseForEvaluationPrompt), jsonOptions)!;
            var score = parsedResponse.Scores.FirstOrDefault();

            if (score == null)
            {
                numericMetric.AddDiagnostic(EvaluationDiagnostic.Error("Score was inconclusive"));
            }
            else
            {
                numericMetric.Value = score.ScoreLabel;

                if (!string.IsNullOrWhiteSpace(score.DescriptionOfQuality))
                {
                    numericMetric.AddDiagnostic(EvaluationDiagnostic.Informational(score.DescriptionOfQuality));
                }
            }

            numericMetric.Interpretation = Interpret(numericMetric);

            return new ValueTask();
        }

        internal static EvaluationMetricInterpretation Interpret(NumericMetric metric)
        {
            double score = metric?.Value ?? -1.0;
            EvaluationRating rating = score switch
            {
                1.0 => EvaluationRating.Unacceptable,
                2.0 => EvaluationRating.Poor,
                3.0 => EvaluationRating.Average,
                4.0 => EvaluationRating.Good,
                5.0 => EvaluationRating.Exceptional,
                _ => EvaluationRating.Inconclusive,
            };
            return new EvaluationMetricInterpretation(rating, failed: rating == EvaluationRating.Inconclusive);
        }

        internal static ReadOnlySpan<char> TrimMarkdownDelimiters(string json)
        {
#if NETSTANDARD2_0
        ReadOnlySpan<char> trimmed = json.ToCharArray();
#else
            ReadOnlySpan<char> trimmed = json;
#endif
            trimmed = trimmed.Trim().Trim(['`']); // trim whitespace and markdown characters from beginning and end
                                                  // trim 'json' marker from markdown if it exists
            if (trimmed.Length > 4 && trimmed[0..4].SequenceEqual(['j', 's', 'o', 'n']))
            {
                trimmed = trimmed.Slice(4);
            }

            return trimmed;
        }


    }

    record ScoringResponse(AnswerScore[] Scores);
    record AnswerScore(int Index, int ScoreLabel, string DescriptionOfQuality);
    public class EvalQuestion
    {
        public required int QuestionId { get; set; }
        public required string Question { get; set; }

        public required string Answer { get; set; }
    }
}
