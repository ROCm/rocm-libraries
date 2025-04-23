#include <vector>

// ExternalSourceImageInfo struct - used to pass the image info needed for external source input.
struct ExternalSourceImageInfo {
    unsigned char* file_data;  // Pointer to the file data
    size_t file_read_size;     // Total file size
    unsigned int width;                 // Width of the image
    unsigned int height;                // Height of the image
    unsigned int channels;              // Number of image channels
    unsigned roi_width;        // ROI width of the image
    unsigned roi_height;       // ROI height of the image
};


class ExternalSourceImageReader {
   public:
    enum class Status {
        OK = 0
    };

    //! Feeds file names as an external_source into the reader
    virtual void feed_file_names(const std::vector<std::string> &file_names, size_t num_images, bool eos = false) = 0;

    //! Used for feeding raw data into the reader (mode specified compressed jpegs or raw)
    virtual void feed_data(const std::vector<unsigned char *> &images, const std::vector<size_t> &image_size, ExternalSourceFileMode mode, bool eos = false, const std::vector<unsigned> roi_width = {}, const std::vector<unsigned> roi_height = {}, unsigned int width = 0, unsigned int height = 0, unsigned int channels = 0) = 0;

    virtual ~ExternalSourceImageReader() = default;
};
