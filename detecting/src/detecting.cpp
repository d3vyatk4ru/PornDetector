
#include "detecting.h"

#include <opencv2/opencv.hpp>

#include <torch/script.h>

#include <boost/beast.hpp>

#include <boost/algorithm/string.hpp>

#include <fstream>

#include <stdlib.h>

#include <sys/socket.h>

#include <algorithm>

#include <string>

#define HEIGHT 224
#define WIDTH 224
#define MAGIC_N 1234// size of symbols in STOPWORDS file

#define STOPWORDS "../../requirement/STOPWORDS.txt"

std::string BAD_SYM = "!#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
const std::string ALPH = "абвгдеёжзийклмнопрстуфхцчшщъьыэюя";

std::ostream &operator<<(std::ostream &out, const Probability &prob) {
    out << "[" << 1 - prob.porn << "," << prob.porn << "]"
        << "\n";
    return out;
}

template<typename T>
TorchWrapper<T>::TorchWrapper() {}

template<typename T>
int TorchWrapper<T>::set_threshold(double _threshold) {
    if (_threshold > 0.0 && _threshold < 1.0) {
        threshold = _threshold;
    } else {
        throw std::logic_error("Threshold can be from 0 to 1");
    }

    return 0;
}

template<typename T>
int TorchWrapper<T>::load_model(const std::string &path) {
    try {
        model = torch::jit::load(path);
        // Error - основной класс ошибок
    } catch (const c10::Error &ex) {
        std::cerr << "Error loading the model\n";
        return 1;
    }

    std::cout << "Successful load DL model\n";
    return 0;
}

PornImageDetector::PornImageDetector(const std::string &path_to_model) {
    load_model(path_to_model);
}

PornImageDetector::~PornImageDetector() {}

Probability PornImageDetector::forward(torch::Tensor &img) {
    torch::NoGradGuard no_grad;// turn off trainable function

    torch::Tensor output = model.forward({img}).toTensor();

    // вектор вероятностей принадлежности классам size = 2
    std::tuple<torch::Tensor, torch::Tensor> result = torch::max(torch::softmax(output, 1), 1);

    torch::Tensor proba_ = std::get<0>(result);
    torch::Tensor index = std::get<1>(result);

    auto proba = proba_.accessor<float, 1>();
    auto idx = index.accessor<long, 1>();

    if (!idx[0]) {
        Probability probability(1 - proba[0]);
        prob.porn = 1 - proba[0];
        return probability;
    } else {
        Probability probability(proba[0]);
        prob.porn = proba[0];
        return probability;
    }
}

std::string PornImageDetector::blurring() {
    if (prob.porn > threshold) {
        cv::GaussianBlur(orig_img, orig_img,
                         cv::Size(31, 31),
                         0);
    }

    return mat2base64(orig_img);
}

void PornImageDetector::permutation_channels(cv::Mat &img) {
    // каналы изображения
    switch (img.channels()) {
        case 4:
            cv::cvtColor(img, img, cv::COLOR_BGRA2RGB);
            break;

        case 3:
            cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
            break;

        case 1:
            cv::cvtColor(img, img, cv::COLOR_GRAY2RGB);
            break;

        default:
            throw std::runtime_error("Incorrect image depth!");
    }
}

cv::Mat PornImageDetector::base642mat(const std::string &base64_code) {
    std::string dest;

    // кодирование base64 -> jpg -> Mat
    dest.resize(boost::beast::detail::base64::decoded_size(base64_code.size()));
    auto const result = boost::beast::detail::base64::decode(&dest[0], base64_code.data(), base64_code.size());
    dest.resize(result.first);

    std::vector<uchar> data(dest.begin(), dest.end());
    return cv::imdecode(cv::Mat(data), 1);
}

std::string PornImageDetector::mat2base64(const cv::Mat &img) {
    std::string dest;

    // кодирование Mat -> jpg -> base64
    std::vector<unsigned char> buffer;
    cv::imencode(".jpg", img, buffer);
    dest.resize(boost::beast::detail::base64::encoded_size(buffer.size()));
    boost::beast::detail::base64::encode(&dest[0], buffer.data(), buffer.size());

    return dest;
}

cv::Mat PornImageDetector::load_img(const std::string &base64_code) {
    // с этим объектом будут происходить манипуляции
    cv::Mat img = base642mat(base64_code);

    // этот объект блюрим --- оригинал
    orig_img = base642mat(base64_code);

    try {
        // BGR to RGB
        permutation_channels(img);
    } catch (const cv::Exception &ex) {
        // если не получилось, работаем с BGR
        std::cout << "Using BGR format for image\n";
    }

    return img;
}

torch::Tensor PornImageDetector::preproccesing(cv::Mat &img) {
    // так как вход сети 224 х 224 кропаем изображение
    cv::Size target_size(WIDTH, HEIGHT);
    cv::resize(img, img, target_size);

    img.convertTo(img, CV_32FC3, 1 / 255.0);

    torch::Tensor img_tensor = torch::from_blob(img.data,
                                                {img.rows, img.cols, img.channels()},
                                                c10::kFloat);

    img_tensor = img_tensor.permute({2, 0, 1});

    // добавляем фиктивную ось в тензор (will be dim = 4)
    img_tensor.unsqueeze_(0);

    return img_tensor.clone();
}

PornTextDetector::PornTextDetector() {

    sock = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);

    if (sock < 0) {
        perror("socket (PornTextDetector)");
        exit(EXIT_FAILURE);
    }

    struct sockaddr_in servername;
    init_sockaddr(&servername, "localhost", 9090);

    if (0 > connect(sock, (struct sockaddr *) &servername, sizeof(servername))) {
        perror("connect (PornTextDetector)");
        exit(EXIT_FAILURE);
    }
}

PornTextDetector::~PornTextDetector() {
    close(sock);
}

void PornTextDetector::init_sockaddr(struct sockaddr_in *name, const char *hostname, uint16_t port) {
    name->sin_family = AF_INET;
    name->sin_port = htons(port);

    struct hostent *hostinfo;
    hostinfo = gethostbyname(hostname);

    if (hostinfo == NULL) {
        std::cout << stderr << "Unknown host " << hostname << "\n";
        exit(EXIT_FAILURE);
    }

    name->sin_addr = *(struct in_addr *) hostinfo->h_addr;
}

void PornTextDetector::write_to_server(int fd, std::string msg) {

    size_t left = msg.size();
    ssize_t sent = 0;

    int flags = 0;
    while (left > 0) {
        sent = ::send(fd, msg.data() + sent, msg.size() - sent, flags);
        if (-1 == sent) {
            throw std::runtime_error("write failed: " + std::string(strerror(errno)));
        }

        left -= sent;
    }
}

void PornTextDetector::read_from_server(int fd, char *buf) {
    int n = {};

    n = ::recv(fd, buf, 2048, MSG_NOSIGNAL);

    std::cout << n << "\n";

    if (-1 == n && errno != EAGAIN) {
        throw std::runtime_error("read failed: " + std::string(strerror(errno)));
    }

    if (0 == n) {
        throw std::runtime_error("client: " + std::to_string(fd) + " disconnected");
    }

    if (-1 == n) {
        throw std::runtime_error("client: " + std::to_string(fd) + " timeouted");
    }
}

std::vector<std::string> PornTextDetector::get_stopwords() {

    std::ifstream fin(STOPWORDS);// открыли файл для чтения
    char buffer[MAGIC_N];
    fin.getline(buffer, MAGIC_N);
    fin.close();

    std::string tmp = buffer;
    std::vector<std::string> stopwords;
    boost::split(stopwords, tmp, [](char c) { return c == ' '; });
    return stopwords;
}

void PornTextDetector::remove_bad_syms_words(Token &token) {
    Token stopwords = get_stopwords();
    Token good_words;
    bool flag = false;

    for (size_t i = 0; i < token.size(); ++i) {
        flag = false;
        for (int j = 0; j < BAD_SYM.size(); ++j) {
            token[i].erase(std::remove(token[i].begin(),
                                       token[i].end(),
                                       BAD_SYM[j]),
                           token[i].end());
        }

        for (size_t j = 0; j < stopwords.size(); ++j) {
            if (token[i] == stopwords[j]) {
                flag = true;
                break;
            }
        }

        if (!flag) {
            good_words.push_back(token[i]);
        }
    }

    token = good_words;
}

std::string PornTextDetector::preproccesing(std::string &text) {

    msg = text;

    //  нижний регистр
    boost::algorithm::to_lower(text);

    //токенизация
    Token token;

    boost::split(token,
                 text,
                 [](char c) { return c == ' '; });

    remove_bad_syms_words(token);

    std::string clear_msg = "";

    for (std::string word : token) {
        clear_msg += word + ' ';
    }

    boost::trim(clear_msg);

    return clear_msg;
}

void PornTextDetector::get_word2vec_representation(std::string &text) {
    write_to_server(sock, text);

    char buffer[2048];
    read_from_server(sock, buffer);

    std::string tmp;
    std::stringstream ss(buffer);
    size_t i = 0;

    while (std::getline(ss, tmp, ' ')) {
        sent2vec[0][i] = torch::tensor(std::stod(tmp));
        i++;
    }
}

Probability PornTextDetector::forward(std::string &text) {

    get_word2vec_representation(text);

    torch::NoGradGuard no_grad;// turn off trainable function

    torch::Tensor output = model.forward({sent2vec}).toTensor();

    // вектор вероятностей принадлежности классам size = 2
    std::tuple<torch::Tensor, torch::Tensor> result = torch::max(output, 1);

    torch::Tensor proba_ = std::get<0>(result);
    torch::Tensor index = std::get<1>(result);

    auto proba = proba_.accessor<float, 1>();
    auto idx = index.accessor<long, 1>();

    if (!idx[0]) {
        Probability probability(1 - proba[0]);
        prob.porn = 1 - proba[0];
        return probability;
    } else {
        Probability probability(proba[0]);
        prob.porn = proba[0];
        return probability;
    }
}

std::string PornTextDetector::text_replace(std::string &text) {

    if (prob.porn > threshold) {

        Token token;

        boost::split(token,
                     text,
                     [](char c) { return c == ' '; });

        std::string new_msg = "";
        for (std::string word : token) {
            for (int i = 0; i < word.length() / 2; ++i) {
                new_msg += '*';
            }

            new_msg += ' ';
        }

        boost::trim(new_msg);
        return new_msg;
    }

    return msg;
}
