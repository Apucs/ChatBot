import trax
from trax import layers as tl
from trax.supervised import training
from config import *
from process import Process
from models import Model

#display the model
# temp_model = Model.ReformerLM(mode = 'train')
# print(str(temp_model))
# # free memory
# del temp_model

class Train:
    def __init__(self):
        pass


    def training_loop(self, ReformerLM, train_gen, eval_gen, output_dir="./model/"):
        """
        Args:
            ReformerLM:  the Reformer language model you are building
            train_gen (generator): train data generator.
            eval_gen (generator): Validation generator.
            output_dir (string): Path to save the model output. Defaults to './model/'.

        Returns:
            trax.supervised.training.Loop: Training loop for the model.
        """

        # use the warmup_and_rsqrt_decay learning rate schedule
        lr_schedule = trax.lr.warmup_and_rsqrt_decay(
            n_warmup_steps=1000, max_value=0.01)


        # define the train task
        train_task = training.TrainTask(
            # labeled data
            labeled_data=train_gen,
            # loss layer
            loss_layer=tl.CrossEntropyLoss(),
            # optimizer
            optimizer=trax.optimizers.Adam(0.01),
            # lr_schedule
            lr_schedule=lr_schedule,
            # n_steps
            n_steps_per_checkpoint=10
        )

        # define the eval task
        eval_task = training.EvalTask(
            # labeled data
            labeled_data=eval_gen,
            # metrics
            metrics=[tl.CrossEntropyLoss(), tl.Accuracy()]
        )

        loop = training.Loop(ReformerLM(mode='train', ),
                             train_task,
                             eval_tasks=[eval_task],
                             output_dir=output_dir)
        return loop


p = Process()

train_data, eval_data, train_stream, eval_stream = p.train_test()

# the stream generators will yield (input, target, weights). let's just grab the input for inspection
#inp, _, _ = next(train_stream)

# print the shape. format is (batch size, token length)
#print("input shape: ", inp.shape)

# detokenize the first element
#print(trax.data.detokenize(inp[0], vocab_dir=VOCAB_DIR, vocab_file=VOCAB_FILE))

tr = Train()
loop = tr.training_loop(Model.ReformerLM, train_stream, eval_stream)
# print(loop)
loop.run(10)



