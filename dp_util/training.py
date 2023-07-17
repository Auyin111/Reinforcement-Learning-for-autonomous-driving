import torch


def valid(net, dataloader, criterion,
          device, verbose=True, sf=5):
    correct = 0
    total = 0

    running_loss = 0

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            images = torch.unsqueeze(images, dim=1)
            labels = labels.to(device)

            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            arr_predicted = predicted.detach().cpu().numpy()
            arr_labels = labels.detach().cpu().numpy()[:, 1]
            correct += (arr_predicted == arr_labels).sum().item()

    avg_loss = running_loss / total
    accuracy = 100 * correct // total

    if verbose:
        print(f'Accuracy of the network on the {total} images: {accuracy} % ang avg_loss: {avg_loss:.{sf}f}')

    return avg_loss, accuracy
