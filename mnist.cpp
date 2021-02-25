#define _USE_MATH_DEFINES

#include <iostream>
#include <fstream>
#include <assert.h>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <math.h>
#include <numeric>
#include <iomanip>

int reverseInt(int i);
std::vector<double> getNeuronOutput(std::vector<double> &input, std::vector<std::vector<double>> &neuronsWeights);
std::vector<double> getReconstructedOutput(std::vector<double> &input, std::vector<std::vector<double>> &neuronsWeights);
void learn(std::vector<double> &input, std::vector<double> &output, std::vector<double> &reconstructedInput, std::vector<std::vector<double>> &neuronsWeights);
void showDistribution(std::vector<double> &data, std::string name, int precision);

int main()
{
    assert(16777216 == reverseInt(1));
    std::ifstream file("mnistData/train-images-idx3-ubyte", std::ios::binary);
    std::vector<std::vector<double>> images;
    if (file.is_open())
    {
        int magicNumber = 0;
        file.read((char *)&magicNumber, sizeof(magicNumber));
        magicNumber = reverseInt(magicNumber);
        assert(magicNumber == 2051);
        int numberOfImages = 0;
        file.read((char *)&numberOfImages, sizeof(numberOfImages));
        numberOfImages = reverseInt(numberOfImages);
        assert(numberOfImages == 60000);
        int rows = 0;
        file.read((char *)&rows, sizeof(rows));
        rows = reverseInt(rows);
        assert(rows == 28);
        int cols = 0;
        file.read((char *)&cols, sizeof(cols));
        cols = reverseInt(cols);
        assert(cols == 28);
        for (int i = 0; i < numberOfImages; i++)
        {
            images.push_back(std::vector<double>());
            for (int j = 0; j < rows; j++)
            {
                for (int k = 0; k < cols; k++)
                {
                    unsigned char pixel = 0;
                    file.read((char *)&pixel, sizeof(pixel));
                    images[i].push_back((double)pixel);
                }
            }
        }
        file.close();
    }
    std::vector<std::string> chars = {" ", ".", "+", "*", "o", "O", "=", "#"};

    for (int row = 0; row < 28; row++)
    { //remove later
        for (int col = 0; col < 28; col++)
        {
            std::cout << chars[((int)images[0][28 * row + col]) / 32] << " ";
        }
        std::cout << std::endl;
    }

    for (std::vector<double> &image : images)
    {
        double mean = 0;
        for (double &pixel : image)
        {
            mean += pixel;
        }
        mean /= image.size();
        for (double &pixel : image)
        {
            pixel -= mean;
        }
        double standardDeviation = 0;
        for (double &pixel : image)
        {
            standardDeviation += pow((pixel - mean), 2);
        }
        standardDeviation = sqrt((standardDeviation / (image.size() - 1)));
        for (double &pixel : image)
        {
            pixel /= standardDeviation;
            //std::cout << "pixel: " << pixel << std::endl;
        }
    }

    std::ifstream labelFile("mnistData/train-labels-idx1-ubyte", std::ios::binary);
    std::vector<double> labels;
    if (labelFile.is_open())
    {
        int magicNumber = 0;
        labelFile.read((char *)&magicNumber, sizeof(magicNumber));
        magicNumber = reverseInt(magicNumber);
        assert(magicNumber == 2049);
        int numberOfItems = 0;
        labelFile.read((char *)&numberOfItems, sizeof(numberOfItems));
        numberOfItems = reverseInt(numberOfItems);
        assert(numberOfItems == 60000);
        for (int i = 0; i < numberOfItems; i++)
        {
            unsigned char label = 0;
            labelFile.read((char *)&label, sizeof(label));
            labels.push_back((double)label);
        }
        labelFile.close();
    }
    std::cout << labels[0] << std::endl; //remove later

    int inputAmount = (28 * 28) + 10;
    std::vector<std::vector<double>> neuronsWeights(inputAmount, std::vector<double>(inputAmount));
    double lower_bound = -0.1;
    double upper_bound = 0.1;
    std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
    std::default_random_engine re;
    for (std::vector<double> &neuron : neuronsWeights)
    {
        for (double &value : neuron)
        {
            value = unif(re);
            //std::cout << "weight value:" << value << std::endl;
        }
        //std::cout << "max: " << *max_element(neuron.begin(), neuron.end()) << std::endl;
    }
    /*for (std::vector<double> &neuron : neuronsWeights)
    {
        showDistribution(neuron, "weights", 4);
    }*/

    std::vector<std::vector<double>> labelsEncoding;
    for (double &label : labels)
    {
        std::vector<double> labelEncoding;
        for (int i = 0; i < 10; i++)
        {
            if (i == label)
            {
                labelEncoding.push_back(1);
            }
            else
            {
                labelEncoding.push_back(0);
            }
        }
        labelsEncoding.push_back(labelEncoding);
    }

    for (double &num : labelsEncoding[0])
    { //remove later
        std::cout << num;
    }
    std::cout << std::endl;

    std::vector<std::vector<double>> inputs;
    for (int i = 0; i < images.size(); i++)
    {
        std::vector<double> temp(images[i]);
        temp.insert(temp.end(), labelsEncoding[i].begin(), labelsEncoding[i].end());
        inputs.push_back(temp);
    }
    assert(images[0].size() == 28 * 28);
    assert(labelsEncoding[0].size() == 10);
    assert(images[0].size() + labelsEncoding[0].size() == inputAmount);
    assert(inputs[0].size() == inputAmount);

    double correct = 0;
    double wrong = 0;

    for (std::vector<double> &inputInt : inputs)
    {
        std::vector<double> inputDouble(inputInt.begin(), inputInt.end());
        std::vector<double> activationOutput = getNeuronOutput(inputDouble, neuronsWeights);
        std::vector<double> reconstructedInput = getReconstructedOutput(activationOutput, neuronsWeights);
        learn(inputDouble, activationOutput, reconstructedInput, neuronsWeights);
        std::cout << "input enconding:";
        std::vector<double> original;
        for (auto it = inputDouble.end() - 10; it != inputDouble.end(); ++it)
        {
            std::cout << " " << *it;
            original.push_back(*it);
        }
        std::cout << std::endl;

        std::cout << "activation output of encoding:";
        for (auto it = activationOutput.end() - 10; it != activationOutput.end(); ++it)
        {
            std::cout << " " << *it;
        }
        std::cout << std::endl;

        std::vector<double> rec;
        std::cout << "reconstructed enconding:";
        for (auto it = reconstructedInput.end() - 10; it != reconstructedInput.end(); ++it)
        {
            std::cout << " " << *it;
            rec.push_back(*it);
        }
        std::cout << std::endl;

        auto originalMax = std::max_element(original.begin(), original.end());
        int originalMaxIndex = originalMax - original.begin();
        auto recMax = std::max_element(rec.begin(), rec.end());
        int recMaxIndex = recMax - rec.begin();
        if (originalMaxIndex == recMaxIndex)
        {
            correct += 1;
        }
        else
        {
            wrong += 1;
        }

        std::cout << "correct: " << correct << " | ";
        std::cout << "wrong: " << wrong << " | ";
        std::cout << "success rate: " << (correct / (correct + wrong)) * 100 << "%" << std::endl;

        std::cout << std::endl;
        //break;
    }
}

int reverseInt(int i)
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

std::vector<double> getNeuronOutput(std::vector<double> &input, std::vector<std::vector<double>> &neuronsWeights)
{
    std::vector<double> output;
    for (std::vector<double> &neuronWeights : neuronsWeights)
    {
        double activity = 0;
        for (int i = 0; i < neuronWeights.size(); i++)
        {
            activity += neuronWeights[i] * input[i];
            //std::cout << neuronWeights[i] << "*" << input[i] << " " << std::endl;
        }
        //std::cout << "activity: " << activity << std::endl;
        double sigmoidResult = ((1.0 / (1.0 + exp(-activity))) - 0.5) * 10;
        output.push_back(activity);
    }
    return output;
}

std::vector<double> getReconstructedOutput(std::vector<double> &input, std::vector<std::vector<double>> &neuronsWeights)
{
    std::vector<double> output(neuronsWeights[0].size(), 0);
    for (int i = 0; i < output.size(); i++)
    {
        double activity = 0;
        for (int j = 0; j < input.size(); j++)
        {
            activity += neuronsWeights[j][i] * input[j];
        }
        double sigmoid = ((1.0 / (1.0 + exp(-activity))) - 0.5) * 10;
        output[i] = activity;
    }
    return output;
}

void learn(std::vector<double> &input, std::vector<double> &output, std::vector<double> &reconstructedInput, std::vector<std::vector<double>> &neuronsWeights)
{
    for (int i = 0; i < output.size(); i++)
    {
        for (int j = 0; j < input.size(); j++)
        {
            neuronsWeights[i][j] += 0.001 * ((output[i] * input[j]) - (output[i] * reconstructedInput[j]));
            /*std::cout << "output:" << output[i] << std::endl;
            std::cout << "input:" << input[j] << std::endl;
            std::cout << "reconstructedInput:" << reconstructedInput[j] << std::endl;*/
        }
    }
}

void showDistribution(std::vector<double> &data, std::string name, int precision)
{
    double mean = std::accumulate(data.begin(), data.end(), 0) / data.size();
    double standardDeviation = 0;
    for (const double &point : data)
    {
        standardDeviation += pow((point - mean), 2);
    }
    standardDeviation = sqrt((standardDeviation / (data.size() - 1)));

    std::vector<std::string> symbols = {"_", "\u2581", "\u2582", "\u2583", "\u2584", "\u2585", "\u2586", "\u2587"};
    int maxRanges = 30;
    double maxElement = *std::max_element(data.begin(), data.end());
    double step = maxElement / maxRanges;
    std::vector<int> distribution(maxRanges, 0);

    for (const double &point : data)
    {
        int index = 0;
        if (point != 0.0)
        {
            index = point / step;
            if (index >= maxRanges)
            {
                index = maxRanges - 1;
            }
        }
        distribution[index] += 1;
    }

    double distributionMax = *std::max_element(distribution.begin(), distribution.end());
    double divider = distributionMax / (symbols.size() - 1);
    std::cout << std::setprecision(precision) << std::fixed;
    std::cout << name << ":\t\t";
    std::cout << "mean:" << mean << "\t\t";
    std::cout << "standard deviation:" << standardDeviation << "\t\t";
    std::cout << "graph: [";
    for (const int &value : distribution)
    {
        std::cout << symbols[value / divider];
    }
    std::cout << "]" << std::defaultfloat << std::endl;
}
