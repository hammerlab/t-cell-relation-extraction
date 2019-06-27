

def train(model, X_train, Y_train, n_epochs=25, lr=0.01, batch_size=32, seed=1):
    """
    Args:
        X_train: Array of numpy arrays containing each token index (each row can vary in length)
        Y_train: 1D numpy array of labels (0 or 1)
    """
    random_state = np.random.RandomState(seed=seed)
    n = len(X_train)
    train_idxs = np.arange(n)

#         self.build_model(**kwargs)
#         self.check_model(lr)

    # Run mini-batch SGD
    st = time()
    for epoch in range(n_epochs):

        # Shuffle training data
        train_idxs = random_state.permutation(list(range(n)))
        Y_train = Y_train[train_idxs]
        X_train = X_train[train_idxs]
        batch_size = min(batch_size, n) 
        epoch_losses = []

        nn.Module.train(self)
        for batch in range(0, n, batch_size):

            # zero gradients for each batch
            self.optimizer.zero_grad()

            if batch_size > len(X_train[batch:batch+batch_size]):
                batch_size = len(X_train[batch:batch+batch_size])

            output = self._pytorch_outputs(X_train[batch:batch + batch_size], None)

            #Calculate loss
            calculated_loss = self.loss(output, torch.Tensor(Y_train[batch:batch+batch_size]))

            #Compute gradient
            calculated_loss.backward()

            #Step on the optimizer
            self.optimizer.step()

            epoch_losses.append(calculated_loss)

            msg = "Epoch {} ({:.2f}s)\tAverage loss={:.6f}, Current loss={:.6f}".format(
                epoch+1, time() - st, torch.stack(epoch_losses).mean(), calculated_loss)
            print(msg)
    print('Training complete')