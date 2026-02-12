#pragma once

#include <windows.h>
#include <combaseapi.h>

////////////////////////////////////////////////////////////////////////////////
// ComUtils.

// A more fully-featured version of ATL's CComPtr, a refcounting smart pointer for managing a COM interface.  Use when
// you have some T* extending IUnknown and don't want to manually refcount.  It can either borrow or steal an existing
// ref, depending on the tag passed to the constructor.
template<typename T>
class ComPtr
{
public:
    struct borrow_tag {};
    struct steal_tag {};

    ComPtr() : m_ptr(nullptr) {}
    ComPtr(std::nullptr_t) : m_ptr(nullptr) {}
    ComPtr(T* ptr, borrow_tag) : m_ptr(ptr) { incref(); }
    ComPtr(T* ptr, steal_tag) : m_ptr(ptr) {}
    ComPtr(const ComPtr<T>& other) : m_ptr(other.m_ptr) { incref(); }
    ComPtr(ComPtr<T>&& other) : m_ptr(other.release()) {}
    ~ComPtr() { decref(); }

    T* get() const { return m_ptr; }
    T* release();

    ComPtr<T>& operator=(ComPtr<T> rhs);

    explicit operator T*() const { return m_ptr; }
    template<typename B> operator ComPtr<B>() const;
    operator bool() const { return (m_ptr != nullptr); }

    T* operator->() const { return m_ptr; }
    T& operator*() const { return *m_ptr; }
    T** operator&() { return &m_ptr; }

private:
    void incref();
    void decref();

    T* m_ptr;
};

// Convenience functions to borrow/steal a COM reference.
template<typename T> inline ComPtr<T> com_borrow(T* ptr) { return ComPtr<T>(ptr, ComPtr<T>::borrow_tag{}); }
template<typename T> inline ComPtr<T> com_steal(T* ptr) { return ComPtr<T>(ptr, ComPtr<T>::steal_tag{}); }


// An RAII utility for managing memory allocated with CoTaskMemAlloc(), taking ownership of the pointer.  Use this with
// interface functions that allocate to avoid having to remember to call CoTaskMemFree() on all codepaths.
template<typename T>
struct ComMemory
{
    ComMemory() : m_ptr(nullptr) {}
    ComMemory(T* ptr) : m_ptr(ptr) {}
    ~ComMemory() { if (m_ptr) { CoTaskMemFree(m_ptr); } }

    T* get() const { return m_ptr; }

    explicit operator T*() const { return m_ptr; }
    operator bool() const { return (m_ptr != nullptr); }

    T* operator->() const { return m_ptr; }
    T& operator*() const { return *m_ptr; }
    T** operator&() { return &m_ptr; }

private:
    T* m_ptr;
};

// Convenience typedefs for commonly used memory types.
using ComString  = ComMemory<char>;
using ComWString = ComMemory<wchar_t>;



////////////////////////////////////////////////////////////////////////////////
// Implementation.

template<typename T>
T* ComPtr<T>::release()
{
  T* ptr = m_ptr;
  m_ptr = nullptr;

  return ptr;
}

template<typename T>
ComPtr<T>& ComPtr<T>::operator=(ComPtr<T> rhs)
{
  decref();
  m_ptr = rhs.release();

  return *this;
}

template<typename T>
template<typename B>
ComPtr<T>::operator ComPtr<B>() const
{
  static_assert(std::is_base_of_v<B, T>);
  return ComPtr<B>(m_ptr, ComPtr<B>::borrow_tag{});
}

template<typename T>
void ComPtr<T>::incref()
{
  if (m_ptr) {
    m_ptr->AddRef();
  }
}

template<typename T>
void ComPtr<T>::decref()
{
  if (m_ptr) {
    m_ptr->Release();
  }
}
