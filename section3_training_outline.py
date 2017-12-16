
# Actually trains this thing
def train_model(use_lstm=True):
    if use_lstm:
        print_and_write("Training the LSTM model with the GPU:" if USE_GPU else "Training the LSTM model:")
    else:
        print_and_write("Training the CNN model with the GPU:" if USE_GPU else "Training the CNN model")

    get_id_to_text()
    embeddings = get_word_embeddings()
    '''
    model_Feature_Extractor = LSTMQA(embeddings) if use_lstm else CNN_Feature_Extractor(embeddings)
    if USE_GPU:
        model.cuda(GPU_NUM)
    '''
    model_Feature_Extractor = CNN_Feature_Extractor(embeddings)
    model_Domain_Classifier = NN_Domain_Classifier()

    #domain classifier loss
    L_d_function = nn.MultiMarginLoss(margin=0.2) #binomial cross entropy loss
    L_y_function = nn.MultiMarginLoss(margin=0.2) #logistic regression loss
    
    optimizer_L_d = optim.Adam(filter(lambda x: x.requires_grad, model_Domain_Classifier.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    optimizer_L_f = optim.Adam(filter(lambda x: x.requires_grad, model_Feature_Extractor.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    orig_time = time()

    for epoch in range(NUM_EPOCHS):
        samples = get_training_data() # recalculate this every epoch to get new random selections
        num_samples = len(samples)

        num_batches = int(math.ceil(1. * num_samples / BATCH_SIZE))
        total_loss = 0 # used for debugging
        for i in range(num_batches):
            # Get the samples ready
            batch = samples[i * BATCH_SIZE: (i+1) * BATCH_SIZE]
            # If this is the last batch, then need to pad the batch to get the same shape as expected
            if i == num_batches - 1 and num_samples % BATCH_SIZE != 0:
                batch = np.concatenate((batch, np.full(((i+1) * BATCH_SIZE - num_samples, 22), "0")), axis=0)

            ########### Set up the feature extractor network ###########
            # Convert from numpy arrays to tensors
            title_tensor, title_lengths = get_tensor_from_batch(batch, use_title=True)
            body_tensor, body_lengths = get_tensor_from_batch(batch, use_title=False)

            # Reset the model
            optimizer_L_d.zero_grad() # where should these be reset?
            optimizer_L_y.zero_grad()

            model_Feature_Extractor.hidden = model_Feature_Extractor.init_hidden()
            # Run our forward pass and get the entire sequence of hidden states
            title_hidden = model_Feature_Extractor(title_tensor)
            title_encoding = get_encodings(title_hidden, title_lengths, use_lstm=use_lstm)
            model_Feature_Extractor.hidden = model_Feature_Extractor.init_hidden()
            body_hidden = model_Feature_Extractor(body_tensor)
            body_encoding = get_encodings(body_hidden, body_lengths, use_lstm=use_lstm)
            mean_hidden_state = (title_encoding + body_encoding) / 2.

            ########### Set up the domain classifier network ###########

            model_Domain_Classifier.hidden = model_Domain_classifier.init_hidden()
            Domain_Classifier_hidden=model_Domain_Classifier(mean_hidden_state)
  
            L_d_loss = L_d_function(Domain_Classifier_hidden,y_d) # get y_d from somewhere?
            total_L_d_loss += L_y_loss.data[0]



            ########### Set up the label predictor network ###########

            # Compute loss, gradients, update parameters
            X_y, y_y = generate_score_matrix(mean_hidden_state)

            L_y_loss = L_y_function(X_y,y_y)
            total_L_y_loss += L_y_loss.data[0]


            #L_d_loss.backward()
            #L_y_loss.backward()
            (L_y-lambda1*L_d).backward()
            optimizer_L_d.step()
            optimizer_L_f.step()

            total_loss += loss.data[0]


            # every so while, check the dev accuracy
            # if i % 10 == 0:
            #     print_and_write("For batch number " + str(i) + " it has taken " + str(time() - orig_time) + " seconds and has loss " + str(total_loss))
            # if i > 0 and i % 100 == 0:
            #     evaluate_model(model, use_lstm=use_lstm)
        print_and_write("For epoch number " + str(epoch) + " it has taken " + str(time() - orig_time) + " seconds and has loss " + str(total_loss))
        evaluate_model(model, use_lstm=use_lstm)
        evaluate_model(model, use_test_data=True, use_lstm=use_lstm)
        if SAVE_MODELS:
            save_checkpoint(epoch, model, optimizer, use_lstm)
    return model
