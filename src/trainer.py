import torch

def trainer(model, device, train_loader, optimizer, loss_fn):
    
    # Training loop
    num_epochs = 30
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            pred = model(images)

            # Compute loss
            loss, loss_items = loss_fn(pred, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}] - Loss: {running_loss/len(train_loader):.4f}')

    # Save the fine-tuned model
    torch.save(model.state_dict(), 'fine_tuned_yolov5s.pth')
