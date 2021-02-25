#include <iostream>
#include <fstream>

#include <armadillo>

using namespace arma;
using namespace std;

// converts from high endian to low endian
int reverseInt(int i)
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void show_distribution(colvec &data)
{
    double mean = arma::mean(data);
    double stddev = arma::stddev(data);
    uvec histogram = hist(data, 50);
    std::vector<std::string> symbols = {"_", "\u2581", "\u2582", "\u2583", "\u2584", "\u2585", "\u2586", "\u2587", "\u2588"};
    int divider = std::max((int)std::ceil((float)max(histogram) / 8), 1);
    cout << "mean: " << mean << "\tstddev: " << stddev << "\t[";
    for(auto it = histogram.begin(); it != histogram.end(); ++it)
    {
        int index = (*it) / divider;
        cout << symbols[index];
    }
    cout << "]" << endl << endl;
}

int main()
{
    arma_rng::set_seed(10);
    // get the data of the image file from the mnist dataset
    mat images;
    ifstream imageFile("mnistData/train-images-idx3-ubyte", std::ios::binary);
    if (imageFile.is_open())
    {
        int magicNumber = 0;
        imageFile.read((char *)&magicNumber, sizeof(magicNumber));
        magicNumber = reverseInt(magicNumber);
        int numberOfImages = 0;
        imageFile.read((char *)&numberOfImages, sizeof(numberOfImages));
        numberOfImages = reverseInt(numberOfImages);
        int rows = 0;
        imageFile.read((char *)&rows, sizeof(rows));
        rows = reverseInt(rows);
        int cols = 0;
        imageFile.read((char *)&cols, sizeof(cols));
        cols = reverseInt(cols);
        images.zeros(numberOfImages, (rows * cols));
        for (int i = 0; i < numberOfImages; ++i)
        {
            for (int j = 0; j < (rows * cols); ++j)
            {
                unsigned char pixel = 0;
                imageFile.read((char *)&pixel, sizeof(pixel));
                images(i, j) = (double)pixel;
            }
        }
        imageFile.close();
    }

    // whitening of the image data
    // for each pixel -> pixel - mean
    // for each pixel -> pixel - stddeviation

    colvec means = mean(images, 1);
    images -= repmat(means, 1, 784);

    colvec stddevs = stddev(images, 0, 1);
    images /= repmat(stddevs, 1, 784);

    // get the data of the label file from the mnist dataset
    colvec labels;
    ifstream labelFile("mnistData/train-labels-idx1-ubyte", std::ios::binary);
    if (labelFile.is_open())
    {
        int magicNumber = 0;
        labelFile.read((char *)&magicNumber, sizeof(magicNumber));
        magicNumber = reverseInt(magicNumber);
        int numberOfItems = 0;
        labelFile.read((char *)&numberOfItems, sizeof(numberOfItems));
        numberOfItems = reverseInt(numberOfItems);
        labels.zeros(numberOfItems);
        for (int i = 0; i < numberOfItems; ++i)
        {
            unsigned char label = 0;
            labelFile.read((char *)&label, sizeof(label));
            labels(i) = (double)label;
        }
        labelFile.close();
    }

    // convert labels to One Hot enconding
    mat oneHot(labels.n_elem, 10, fill::zeros);
    for (int i = 0; i < labels.n_elem; ++i)
    {
        int index = labels(i);
        oneHot(i, index) = 1;
    }

    // insert One Hot enconding to input
    images.insert_cols(images.n_cols, oneHot);

    // set neuron amount
    int neuronAmount = 1000;

    // initialize weights
    mat weights(images.n_cols, neuronAmount, fill::randu);
    weights /= 100;
    weights += 0.1;

    // start learning
    int correct = 0;
    int wrong = 0;
    for (int i = 0; i < images.n_rows; i++)
    {
        rowvec image = images.row(i);
        mat neurons = image * weights;
        rowvec recImage = neurons * weights.t();
        weights += 0.001 * ((neurons.t() * image) - (neurons.t() * recImage)).t();
        if (index_max(image.tail(10)) == index_max(recImage.tail(10)))
        {
            ++correct;
        }
        else
        {
            ++wrong;
        }

        colvec colimage = image.t();
        colvec colweights = vectorise(weights);
        colvec colneurons = vectorise(neurons);
        colvec colrecImage = recImage.t();

        // cout << "image data:" << endl;
        // show_distribution(colimage);
        // cout << "weights data:" << endl;
        // show_distribution(colweights);
        // cout << "neuron activity:" << endl;
        // show_distribution(colneurons);
        // cout << "rec image data:" << endl;
        // show_distribution(colrecImage);

        cout << "input encoding:\t";
        image.tail(10).print();
        cout << "reconstructed input enconding:\t";
        recImage.tail(10).print();

        cout << "correct: " << correct << " | ";
        cout << "wrong: " << wrong << " | ";
        cout << "success rate: " << ((double)correct / (correct + wrong)) * 100 << "%" << endl;
    }
}