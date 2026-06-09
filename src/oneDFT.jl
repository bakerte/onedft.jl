module oneDFT

  """
    libdir

  Shows where the One-DFT is in the filesystem
  """
  const libdir = @__DIR__

  const libpath = libdir*"/lib/"

  files = ["imports.jl",#="banner.jl",=#"exports.jl"]
  for w = 1:length(files)
    include(libpath*files[w])
  end

  files = ["lda.jl","exchangecorrelation.jl"]
  subdir = "lda/"
  for w = 1:length(files)
    include(libpath*subdir*files[w])
  end

  files = ["findvKS.jl"]
  subdir = "exactKS/"
  for w = 1:length(files)
    include(libpath*subdir*files[w])
  end

end

using .oneDFT
